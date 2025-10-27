# -*- coding:utf-8 -*-
import os
import json

import sys
import mne
import torch
import random
import shutil
import argparse
import warnings
import numpy as np
import torch.optim as opt
from tqdm import tqdm
from torch.amp import autocast
from torch.amp import GradScaler

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.utils import model_size, set_random_seed
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from dataset.utils import split_train_test_val_files
from data_loader import EEGDataset
from model.neuronet import NeuroNet

warnings.filterwarnings(action="ignore")

class Trainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.n_fold = fold
        self.config = config
        self.model_name = config["name"]
        self.ds_cfg = config["Dataset"]

        self.fs = self.ds_cfg["rfreq"]
        self.raw_fs = self.ds_cfg["sfreq"]
        self.second = self.ds_cfg["second"]
        self.time_window = self.ds_cfg["time_window"]
        self.time_step = self.ds_cfg["time_step"]
        self.data_scaler = self.ds_cfg["data_scaler"]
        self.base_path = self.ds_cfg["base_path"]
        self.ckpt_path = self.ds_cfg["ckpt_path"]
        self.k_splits = self.ds_cfg["k_splits"]

        self.encoder_embed_dim = self.config["Encoder"]["encoder_embed_dim"]
        self.encoder_heads = self.config["Encoder"]["encoder_heads"]
        self.encoder_depths = self.config["Encoder"]["encoder_depths"]
        self.alpha = self.config["Encoder"]["alpha"]

        self.decoder_embed_dim = self.config["Decoder"]["decoder_embed_dim"]
        self.decoder_heads = self.config["Decoder"]["decoder_heads"]
        self.decoder_depths = self.config["Decoder"]["decoder_depths"]

        self.projection_hidden = self.config["Encoder"]["projection_hidden"]

        self.temperature = self.config["training_params"]["temperature"]
        self.mask_ratio = self.config["training_params"]["mask_ratio"]
        self.print_point = self.config["training_params"]["print_point"]

        self.train_epochs = self.config["training_params"]["train_epochs"]
        self.warmup_epochs = self.config["training_params"]["train_warmup_epoch"]

        self.batch_size = self.config["training_params"]["batch_size"]
        self.train_batch_accumulation = self.config["training_params"][
            "train_batch_accumulation"
        ]
        self.eff_batch_size = self.batch_size * self.train_batch_accumulation

        self.train_base_learning_rate = self.config["training_params"][
            "train_base_learning_rate"
        ]

        self.lr = self.train_base_learning_rate * self.eff_batch_size / 256
        self.model = self.build_model()

        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_epochs
        )
        self.tensorboard_path = os.path.join(
            self.ckpt_path, self.model_name, str(self.n_fold), "tensorboard"
        )

        # remote tensorboard files
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)

        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)
        self.scaler = GradScaler("cuda")  # 添加混合精度缩放器

        print("Model Size : {0:.2f}MB".format(model_size(self.model)))

        print("Frame Size : {}".format(self.num_patches))
        print("Leaning Rate : {0}".format(self.lr))

    def build_model(self):
        model = NeuroNet(
            fs=self.fs,
            second=self.second,
            time_window=self.time_window,
            time_step=self.time_step,
            encoder_embed_dim=self.encoder_embed_dim,
            encoder_heads=self.encoder_heads,
            encoder_depths=self.encoder_depths,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_heads=self.decoder_heads,
            decoder_depths=self.decoder_depths,
            projection_hidden=self.projection_hidden,
            temperature=self.temperature,
        )
        # 保存需要访问的属性
        self.num_patches = model.num_patches  # 73

        print(
            "[INFO] Number of params of model: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        
        # 使用DataParallel进行多GPU训练
        if torch.cuda.is_available():
            gpu_ids = [int(g) for g in self.args.gpu.split(",") if g.strip()]
            if len(gpu_ids) > 1:
                print(f"Using DataParallel on GPUs: {gpu_ids}")
                model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            self.device = torch.device("cuda:0")  # 主设备
        else:
            self.device = torch.device("cpu")
            
        model = model.to(self.device)
        print(f"[INFO] Model prepared, Device used: {self.device}")
        return model

    def train(self):
        print("K-Fold : {}/{}".format(self.n_fold, self.k_splits))
        train_dataset = EEGDataset(self.config, self.n_fold, set="train")
        # 优化DataLoader配置以提高GPU利用率
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,  # 增加数据加载线程数
            pin_memory=True,
            prefetch_factor=2,  # 预取因子
            persistent_workers=True  # 持久化workers
        )
        val_dataset = EEGDataset(self.config, self.n_fold, set="val")
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        eval_dataset = EEGDataset(self.config, self.n_fold, set="test")
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        total_step = 0
        best_model_state, best_score = self.model.state_dict(), 0

        for epoch in range(self.train_epochs):
            step = 0
            self.model.train()
            self.optimizer.zero_grad()

            # 使用tqdm显示训练进度
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.train_epochs}")

            for x, _ in pbar:
                x = x.to(self.device, non_blocking=True)  # 使用non_blocking提高效率
                with autocast(device_type="cuda"):
                    out = self.model(x, mask_ratio=self.mask_ratio)
                    recon_loss, contrastive_loss, (cl_labels, cl_logits) = out

                    if hasattr(recon_loss, "shape") and recon_loss.dim() > 0:
                        recon_loss = recon_loss.mean()
                    if (
                        hasattr(contrastive_loss, "shape")
                        and contrastive_loss.dim() > 0
                    ):
                        contrastive_loss = contrastive_loss.mean()

                    loss = recon_loss + self.alpha * contrastive_loss
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                if (step + 1) % self.train_batch_accumulation == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if (total_step + 1) % self.print_point == 0:
                    acc = self.compute_metrics(cl_logits, cl_labels)
                    pbar.set_postfix(
                        {
                            "Recon Loss": f"{recon_loss.item():.4f}",
                            "Contrast Loss": f"{contrastive_loss.item():.4f}",
                            "Total Loss": f"{loss.item():.4f}",
                            "Acc": f"{acc.item():.4f}",
                        }
                    )

                self.tensorboard_writer.add_scalar(
                    "Reconstruction Loss", recon_loss, total_step
                )
                self.tensorboard_writer.add_scalar(
                    "Contrastive loss", contrastive_loss, total_step
                )
                self.tensorboard_writer.add_scalar("Total loss", loss, total_step)

                step += 1
                total_step += 1

            val_acc, val_mf1 = self.linear_probing(val_dataloader, eval_dataloader)

            if val_mf1 > best_score:
                best_model_state = self.model.state_dict()
                best_score = val_mf1

            print(
                "[Epoch] : {0:03d} \t [Accuracy] : {1:2.4f} \t [Macro-F1] : {2:2.4f} \n".format(
                    epoch, val_acc * 100, val_mf1 * 100
                )
            )
            self.tensorboard_writer.add_scalar(
                "Validation Accuracy", val_acc, total_step
            )
            self.tensorboard_writer.add_scalar(
                "Validation Macro-F1", val_mf1, total_step
            )

            self.scheduler.step()

        self.save_ckpt(model_state=best_model_state)

    def linear_probing(self, val_dataloader, eval_dataloader):
        self.model.eval()
        (train_x, train_y), (test_x, test_y) = self.get_latent_vector(
            val_dataloader
        ), self.get_latent_vector(eval_dataloader)
        pca = PCA(n_components=50)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)

        model = KNeighborsClassifier()
        model.fit(train_x, train_y)

        out = model.predict(test_x)
        acc, mf1 = accuracy_score(test_y, out), f1_score(test_y, out, average="macro")
        self.model.train()
        return acc, mf1

    def get_latent_vector(self, dataloader):
        total_x, total_y = [], []
        with torch.no_grad():
            for data in dataloader:
                x, y = data
                x, y = x.to(self.device, non_blocking=True), y.to(
                    self.device, non_blocking=True
                )
                # 正确访问模型的forward_latent方法（DataParallel兼容）
                if hasattr(self.model, 'module'):
                    latent = self.model.module.forward_latent(x)
                else:
                    latent = self.model.forward_latent(x)
                total_x.append(latent.detach().cpu().numpy())
                total_y.append(y.detach().cpu().numpy())
                del latent, x, y
        total_x, total_y = np.concatenate(total_x, axis=0), np.concatenate(
            total_y, axis=0
        )
        return total_x, total_y

    def save_ckpt(self, model_state):
        ckpt_path = os.path.join(
            self.ckpt_path, self.model_name, str(self.n_fold), "model"
        )
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        torch.save(
            {
                "model_name": "NeuroNet",
                "model_state": model_state,
                "model_parameter": {
                    "fs": self.fs,
                    "second": self.second,
                    "time_window": self.time_window,
                    "time_step": self.time_step,
                    "encoder_embed_dim": self.encoder_embed_dim,
                    "encoder_heads": self.encoder_heads,
                    "encoder_depths": self.encoder_depths,
                    "decoder_embed_dim": self.decoder_embed_dim,
                    "decoder_heads": self.decoder_heads,
                    "decoder_depths": self.decoder_depths,
                    "projection_hidden": self.projection_hidden,
                    "temperature": self.temperature,
                },
                "hyperparameter": self.__dict__,
            },
            os.path.join(ckpt_path, "best_model.pth"),
        )

    @staticmethod
    def compute_metrics(output, target):
        output = output.argmax(dim=-1)
        accuracy = torch.mean(torch.eq(target, output).to(torch.float32))
        return accuracy

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=777, help="random seed")
    parser.add_argument("--gpu", type=str, default="0,1,2,3,4,5,6,7", help="gpu id")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/chenlungan/算法模型/NeuroNet/configs/model.json",
        help="config file path",
    )
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config["name"] = os.path.basename(args.config).replace(".json", "")

    # 为每个fold创建trainer并训练
    for fold in range(1, config["Dataset"]["k_splits"] + 1):
        trainer = Trainer(args, fold, config)
        trainer.train()

if __name__ == "__main__":
    main()