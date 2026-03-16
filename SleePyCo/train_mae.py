import os
import json
import argparse
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from utils import *
from loss import SupConLoss
from loader import EEGDataLoader
from models.MAE.sleepMAE import SleepMAE
from models.utils import Conv1d

# train_crl用于预训练对比学习模型


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold

        self.cfg = config
        self.tp_cfg = config["training_params"]
        self.es_cfg = self.tp_cfg["early_stopping"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[INFO] Config name: {}".format(config["name"]))

        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()

        self.alpha = self.cfg["training_params"]["alpha"]
        self.contastive_loss = SupConLoss(temperature=self.tp_cfg["temperature"])
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.tp_cfg["lr"],
            weight_decay=self.tp_cfg["weight_decay"],
        )

        self.ckpt_path = os.path.join("checkpoints", config["name"])
        self.ckpt_name = "ckpt_fold-{0:02d}.pth".format(self.fold)
        self.early_stopping = EarlyStopping(
            patience=self.es_cfg["patience"],
            verbose=True,
            ckpt_path=self.ckpt_path,
            ckpt_name=self.ckpt_name,
            mode=self.es_cfg["mode"],
        )

        self.train_losses = []
        self.val_losses = []  # 记录每个epoch的训练和验证损失

        self.fold_start_time = None
        self.fold_end_time = None  # 添加训练时间记录

    def build_model(self):
        model = SleepMAE(self.cfg)
        print(
            "[INFO] Number of params of model: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        model = torch.nn.DataParallel(
            model, device_ids=list(range(len(self.args.gpu.split(","))))
        )
        model.to(self.device)
        print(
            "[INFO] Model prepared, Device used: {} GPU:{}".format(
                self.device, self.args.gpu
            )
        )

        return model

    def build_dataloader(self):
        dataloader_args = {
            "batch_size": self.tp_cfg["batch_size"],
            "shuffle": True,
            "num_workers": 4 * len(self.args.gpu.split(",")),
            "pin_memory": True,
        }
        train_dataset = EEGDataLoader(self.cfg, self.fold, set="train")
        train_loader = DataLoader(dataset=train_dataset, **dataloader_args)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set="val")
        val_loader = DataLoader(dataset=val_dataset, **dataloader_args)
        print("[INFO] Dataloader prepared")

        return {"train": train_loader, "val": val_loader}

    def train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        train_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict["train"]):  # inputs为大小为2的list，包含两个增强后的样本，labels为标签
            loss = 0
            labels = labels.view(-1).to(self.device)  # B,

            inputs = inputs.to(self.device)  # B,1,3000 
            rec_loss, z1, z2 = self.model(inputs) 
            feature = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
            contrastive_loss = self.contastive_loss(feature, labels)
            loss = rec_loss + contrastive_loss

            if loss.numel() > 1:
                loss = loss.mean()
            loss += loss  

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.train_iter += 1

            progress_bar(i,len(self.loader_dict["train"]),"Lr: %.4e | Loss: %.3f"% (get_lr(self.optimizer), train_loss / (i + 1)))

            if self.train_iter % self.tp_cfg["val_period"] == 0:
                print("")
                val_loss = self.evaluate(mode="val")
                self.early_stopping(None, val_loss, self.model)

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

                # 重置训练损失统计
                train_loss = 0
                num_batches = 0

                self.model.train()
                if self.early_stopping.early_stop:
                    break

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        eval_loss = 0

        for i, (inputs, labels) in enumerate(self.loader_dict[mode]):
            loss = 0
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            inputs = inputs.to(self.device)  # B,1,3000 
            loss, _, _ = self.model(inputs) 
            if loss.numel() > 1:
                loss = loss.mean()

            eval_loss += loss.item()

            progress_bar(
                i,
                len(self.loader_dict[mode]),
                "Lr: %.4e | Loss: %.3f" % (get_lr(self.optimizer), eval_loss / (i + 1)),
            )

        return eval_loss

    def run(self):
        self.fold_start_time = time.time()
        for epoch in range(self.tp_cfg["max_epochs"]):
            print("\n[INFO] Fold: {}, Epoch: {}".format(self.fold, epoch))
            self.train_one_epoch()
            if self.early_stopping.early_stop:
                break

        # 记录训练结束时间
        self.fold_end_time = time.time()
        fold_duration = self.fold_end_time - self.fold_start_time
        # 计算训练时长
        hours, remainder = divmod(fold_duration, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(
            f'[INFO] Fold {self.fold} training completed at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.fold_end_time))}'
        )
        print(
            f"[INFO] Fold {self.fold} training duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        )

        loss_data = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "training_time": fold_duration,
            "training_time_formatted": f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
        }
        os.makedirs(self.ckpt_path, exist_ok=True)
        np.save(
            os.path.join(self.ckpt_path, f"losses_fold-{self.fold:02d}.npy"), loss_data
        )


def determine_start_fold_by_loss(config):
    """
    通过检查loss文件来确定哪些fold需要训练
    返回需要训练的fold列表而不是起始fold
    """
    ckpt_path = os.path.join("checkpoints", config["name"])

    # 如果检查点目录不存在，训练所有folds
    if not os.path.exists(ckpt_path):
        print("[INFO] Checkpoint directory does not exist, training all folds")
        return list(range(1, config["dataset"]["num_splits"] + 1))

    folds_to_train = []
    num_folds = config["dataset"]["num_splits"]

    # 检查每个fold的loss文件是否存在
    for fold in range(1, num_folds + 1):
        loss_file = os.path.join(ckpt_path, f"losses_fold-{fold:02d}.npy")

        # 如果loss文件不存在，需要训练这个fold
        if not os.path.exists(loss_file):
            folds_to_train.append(fold)
            print(f"[INFO] Loss file for fold {fold} not found, will train this fold")
        else:
            print(f"[INFO] Loss file for fold {fold} found, skipping this fold")

    # 如果所有fold都已完成，返回空列表
    if not folds_to_train:
        print("[INFO] All folds have been trained, no training needed")

    return folds_to_train


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpu", type=str, default="0,1,2,3,4,5,6,7", help="gpu id")
    parser.add_argument("--config", type=str, default="/home/chenlungan/算法模型/SleePyCo/configs/MAE/SleePyCo-Transformer_SL-01_numScales-1_Sleep-EDF-2018_pretrainMAE.json",help="config file path")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:

        config = json.load(config_file)
    config["name"] = os.path.basename(args.config).replace(".json", "")

    # 确定需要训练的folds
    folds_to_train = determine_start_fold_by_loss(config)
    print(f"[INFO] Folds to train: {folds_to_train}")

    # 只训练缺失的folds
    for fold in folds_to_train:
        print(f"[INFO] Starting training for fold {fold}")
        trainer = OneFoldTrainer(args, fold, config)
        trainer.run()


if __name__ == "__main__":
    main()
