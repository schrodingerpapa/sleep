import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from loader import EEGDataLoader
from utils import EarlyStopping, progress_bar, set_random_seed, summarize_result

from models.CotarFormer import CotarFormer
from models.deepsleepnet import DeepSleepNetFeature
from models.iitnet import IITNetBackbone
from models.sleepcov2 import SleePyCoBackboneV2
from models.sleepcov3 import SleePyCoBackboneV3
from models.sleepyco import SleePyCoBackbone
from models.sleepycolight import SleePyCoLightBackbone
from models.sleepycolightV2 import SleePyCoUltraLightBackbone
from models.sleepycolightV2withoutECA import Light_ECABackbone
from models.sleepycolightV3 import SleePyCoLightV3Backbone
from models.tinysleepnet import TinySleepNetFeature
from models.utime import UTimeEncoder
from models.xsleepnet import XSleepNetFeature


class Seq2SeqEEGDataLoader(EEGDataLoader):
    """Return all labels in a context window instead of labels[target_idx]."""

    def __init__(self, config, fold, set="train", return_metadata=False):
        self.return_metadata = return_metadata
        super().__init__(config, fold, set=set)

    def __getitem__(self, idx):
        file_idx, start_idx, seq_len = self.epochs[idx]
        n_sample = 30 * self.sr * seq_len

        inputs = self.inputs[file_idx][start_idx : start_idx + seq_len]
        inputs = inputs.reshape(1, n_sample)
        inputs = torch.from_numpy(inputs).float()

        labels = self.labels[file_idx][start_idx : start_idx + seq_len]
        labels = torch.from_numpy(labels).long()

        if self.return_metadata:
            metadata = {
                "file_idx": torch.full((seq_len,), file_idx, dtype=torch.long),
                "epoch_idx": torch.arange(start_idx, start_idx + seq_len).long(),
            }
            return inputs, labels, metadata

        return inputs, labels


class GRUSeq2SeqClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config["classifier"]
        self.bidirectional = cfg.get("bidirectional", True)
        self.rnn = nn.GRU(
            input_size=cfg["input_dim"],
            hidden_size=cfg["hidden_dim"],
            num_layers=cfg["num_rnn_layers"],
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        output_dim = cfg["hidden_dim"] * (2 if self.bidirectional else 1)
        dropout = cfg.get("dropout_rate", 0.5 if cfg.get("dropout", False) else 0.0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(output_dim, cfg["num_classes"])

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dropout(x)
        return self.fc(x)


class AttentionGRUSeq2SeqClassifier(nn.Module):
    """Seq2Seq GRU head with the same parameters as classifiers.AttGRU."""

    def __init__(self, config):
        super().__init__()
        cfg = config["classifier"]
        self.bidirectional = cfg.get("bidirectional", True)
        self.rnn = nn.GRU(
            input_size=cfg["input_dim"],
            hidden_size=cfg["hidden_dim"],
            num_layers=cfg["num_rnn_layers"],
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        rnn_out_dim = cfg["hidden_dim"] * (2 if self.bidirectional else 1)
        attn_dim = cfg.get("attention_dim", cfg["hidden_dim"])
        self.w_ha = nn.Linear(rnn_out_dim, attn_dim, bias=True)
        self.w_att = nn.Linear(attn_dim, 1, bias=False)
        dropout = cfg.get("dropout_rate", 0.5 if cfg.get("dropout", False) else 0.0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(attn_dim, cfg["num_classes"])

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        a_states = torch.tanh(self.w_ha(rnn_output))
        alpha = torch.softmax(self.w_att(a_states), dim=1)
        x = a_states * (1.0 + alpha)
        x = self.dropout(x)
        return self.fc(x)


class TransformerSeq2SeqClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config["classifier"]
        self.input_proj = nn.Linear(cfg["input_dim"], cfg["model_dim"])
        layer = nn.TransformerEncoderLayer(
            d_model=cfg["model_dim"],
            nhead=cfg.get("nheads", 8),
            dim_feedforward=cfg.get("feedforward_dim", cfg["model_dim"] * 4),
            dropout=0.1 if cfg.get("dropout", False) else 0.0,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.get("num_encoder_layers", 6)
        )
        self.dropout = nn.Dropout(0.5) if cfg.get("dropout", False) else nn.Identity()
        self.fc = nn.Linear(cfg["model_dim"], cfg["num_classes"])

    def forward(self, x):
        x = self.input_proj(x).transpose(0, 1)
        x = self.encoder(x).transpose(0, 1)
        x = self.dropout(x)
        return self.fc(x)


def build_seq2seq_classifier(config):
    name = config["classifier"]["name"]
    if name in {"GRUSeq2Seq", "PlainGRUSeq2Seq"}:
        return GRUSeq2SeqClassifier(config)
    if name == "AttentionGRUSeq2Seq":
        return AttentionGRUSeq2SeqClassifier(config)
    if name == "TransformerSeq2Seq":
        return TransformerSeq2SeqClassifier(config)
    raise NotImplementedError("seq2seq classifier not supported: {}".format(name))


def build_backbone(config):
    name = config["backbone"]["name"]
    if name == "SleePyCo":
        return SleePyCoBackbone(config)
    if name == "SleePyCoV2":
        return SleePyCoBackboneV2(config)
    if name == "SleePyCoV3":
        return SleePyCoBackboneV3(config)
    if name == "SleePyCoLight":
        return SleePyCoLightBackbone(config)
    if name == "SleePyCoLightV2":
        return SleePyCoUltraLightBackbone(config)
    if name == "SleePyCoLightV3":
        return SleePyCoLightV3Backbone(config)
    if name == "Light_ECA":
        return Light_ECABackbone(config)
    if name == "CotarFormer":
        return CotarFormer(config)
    if name == "XSleepNet":
        return XSleepNetFeature(config)
    if name == "UTime":
        return UTimeEncoder(config)
    if name == "IITNet":
        return IITNetBackbone(config)
    if name == "DeepSleepNet":
        return DeepSleepNetFeature(config)
    if name == "TinySleepNet":
        return TinySleepNetFeature(config)
    raise NotImplementedError("backbone not supported: {}".format(name))


class Seq2SeqModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.epoch_samples = config["dataset"].get("epoch_samples", 3000)
        self.feature = build_backbone(config)
        self.share_classifier_across_scales = config["classifier"].get(
            "share_across_scales", True
        )
        print("[INFO] Seq2Seq classifier: {}".format(config["classifier"]["name"]))
        print(
            "[INFO] Share seq2seq classifier across scales: {}".format(
                self.share_classifier_across_scales
            )
        )
        if self.share_classifier_across_scales:
            self.classifier = build_seq2seq_classifier(config)
        else:
            self.classifiers = nn.ModuleList(
                [
                    build_seq2seq_classifier(config)
                    for _ in range(config["feature_pyramid"]["num_scales"])
                ]
            )

        print(
            "[INFO] Number of params of backbone: ",
            sum(p.numel() for p in self.feature.parameters() if p.requires_grad),
        )
        print(
            "[INFO] Number of params of seq2seq classifiers: ",
            self.count_classifier_params(),
        )

    def count_classifier_params(self):
        if self.share_classifier_across_scales:
            return sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.classifiers.parameters() if p.requires_grad)

    def forward(self, x):
        batch_size, channels, n_samples = x.shape
        if n_samples % self.epoch_samples != 0:
            raise ValueError(
                "Input length {} is not divisible by epoch_samples {}".format(
                    n_samples, self.epoch_samples
                )
            )

        seq_len = n_samples // self.epoch_samples
        epoch_inputs = x.view(
            batch_size, channels, seq_len, self.epoch_samples
        ).transpose(1, 2)
        epoch_inputs = epoch_inputs.reshape(
            batch_size * seq_len, channels, self.epoch_samples
        )

        features = self.feature(epoch_inputs)
        outputs = []
        for scale_idx, feature in enumerate(features):
            pooled = F.adaptive_avg_pool1d(feature, 1).squeeze(-1)
            seq_features = pooled.view(batch_size, seq_len, -1)
            classifier = (
                self.classifier
                if self.share_classifier_across_scales
                else self.classifiers[scale_idx]
            )
            outputs.append(classifier(seq_features))

        return outputs


class OneFoldSeq2SeqTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        self.cfg = config
        self.ds_cfg = config["dataset"]
        self.fp_cfg = config["feature_pyramid"]
        self.tp_cfg = config["training_params"]
        self.es_cfg = self.tp_cfg["early_stopping"]
        self.eval_strategy = self.tp_cfg.get("eval_strategy", "mean_logits")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_ids = [gpu_id for gpu_id in self.args.gpu.split(",") if gpu_id != ""]
        self.use_amp = bool(self.args.amp and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.train_iter = 0

        print("[INFO] Config name: {}".format(config["name"]))
        if self.use_amp:
            print("[INFO] AMP mixed precision enabled")
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        self.criterion = nn.CrossEntropyLoss()
        self.activate_train_mode()
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
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
        self.val_losses = []

    def build_model(self):
        model = Seq2SeqModel(self.cfg)
        print(
            "[INFO] Number of params of model: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        if torch.cuda.is_available() and len(self.gpu_ids) > 1:
            model = torch.nn.DataParallel(
                model, device_ids=list(range(len(self.gpu_ids)))
            )

        if self.tp_cfg["mode"] != "scratch":
            load_path = self.resolve_pretrain_checkpoint()
            print("[INFO] Model loaded for seq2seq finetune: {}".format(load_path))
            state_dict = torch.load(load_path, map_location=self.device)
            state_dict = self.normalize_state_dict_for_model(model, state_dict)
            model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        print(
            "[INFO] Model prepared, Device used: {} GPU:{}".format(
                self.device, self.args.gpu
            )
        )
        return model

    @staticmethod
    def normalize_state_dict_for_model(model, state_dict):
        model_is_parallel = hasattr(model, "module")
        state_is_parallel = any(key.startswith("module.") for key in state_dict)

        if model_is_parallel or not state_is_parallel:
            return state_dict

        return {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    def unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def resolve_pretrain_checkpoint(self):
        explicit_path = self.tp_cfg.get("pretrain_checkpoint")
        if explicit_path:
            if not os.path.isabs(explicit_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                repo_dir = os.path.dirname(script_dir)
                config_dir = self.cfg.get("_config_dir", script_dir)
                candidates = [
                    os.path.abspath(explicit_path),
                    os.path.abspath(os.path.join(config_dir, explicit_path)),
                    os.path.abspath(os.path.join(script_dir, explicit_path)),
                    os.path.abspath(os.path.join(repo_dir, explicit_path)),
                ]
                for candidate in candidates:
                    if os.path.exists(candidate):
                        return candidate
                raise FileNotFoundError(candidates[0])
            return explicit_path

        load_name = self.cfg["name"].replace(self.cfg["classifier"]["name"], "Transformer")
        load_name = load_name.replace(
            "SL-{:02d}".format(self.ds_cfg["seq_len"]), "SL-01"
        )
        load_name = load_name.replace(
            "numScales-{}".format(self.fp_cfg["num_scales"]), "numScales-1"
        )
        load_name = load_name.replace(self.tp_cfg["mode"], "pretrain")
        return os.path.join(
            "checkpoints", load_name, "ckpt_fold-{0:02d}.pth".format(self.fold)
        )

    def build_dataloader(self):
        num_workers = (
            self.args.num_workers
            if self.args.num_workers is not None
            else min(16, max(4, 2 * len(self.gpu_ids)))
        )
        pin_memory = self.device.type == "cuda"

        loaders = {}
        for split in ["train", "val", "test"]:
            dataset = Seq2SeqEEGDataLoader(
                self.cfg,
                self.fold,
                set=split,
                return_metadata=(split != "train"),
            )
            loader_kwargs = {
                "dataset": dataset,
                "batch_size": self.tp_cfg["batch_size"],
                "shuffle": (split == "train"),
                "num_workers": num_workers,
                "pin_memory": pin_memory,
            }
            if num_workers > 0:
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = self.args.prefetch_factor
            loaders[split] = DataLoader(
                **loader_kwargs
            )

        print(
            "[INFO] Seq2Seq dataloaders prepared, num_workers={}, pin_memory={}".format(
                num_workers, pin_memory
            )
        )
        return loaders

    def activate_train_mode(self):
        self.model.train()
        if self.tp_cfg["mode"] != "freezefinetune":
            return

        print("[INFO] Freeze backbone for seq2seq finetune")
        feature = self.unwrap_model().feature
        feature.train(False)
        for param in feature.parameters():
            param.requires_grad = False

        for name in ["conv_c5", "conv_c4", "conv_c3"]:
            if hasattr(feature, name):
                print("[INFO] Unfreeze {}".format(name))
                module = getattr(feature, name)
                module.train(True)
                for param in module.parameters():
                    param.requires_grad = True

    def compute_loss_and_logits(self, inputs, labels):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(inputs)
            logits_sum = torch.zeros_like(outputs[0])
            loss = 0
            flat_labels = labels.reshape(-1)

            for logits in outputs:
                loss = loss + self.criterion(
                    logits.reshape(-1, logits.size(-1)), flat_labels
                )
                logits_sum = logits_sum + logits

        return loss, logits_sum

    def train_one_epoch(self, epoch):
        correct, total, train_loss = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict["train"]):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            total += labels.numel()

            loss, logits = self.compute_loss_and_logits(inputs, labels)

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(logits, dim=-1)
            correct += predicted.eq(labels).sum().item()
            self.train_iter += 1

            progress_bar(
                i,
                len(self.loader_dict["train"]),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (train_loss / (i + 1), 100.0 * correct / total, correct, total),
            )

            if self.train_iter % self.tp_cfg["val_period"] == 0:
                print("")
                val_acc, val_loss = self.evaluate(mode="val")
                self.val_losses.append(val_loss)
                self.early_stopping(val_acc, val_loss, self.model)
                self.activate_train_mode()
                if self.early_stopping.early_stop:
                    break

        if len(self.loader_dict["train"]) > 0:
            self.train_losses.append(train_loss / len(self.loader_dict["train"]))

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg["classifier"]["num_classes"]))
        aggregate_epochs = self.eval_strategy == "mean_logits"
        logit_sums, logit_counts, label_map = {}, {}, {}

        for i, batch in enumerate(self.loader_dict[mode]):
            if len(batch) == 3:
                inputs, labels, metadata = batch
            else:
                inputs, labels = batch
                metadata = None

            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            total += labels.numel()

            loss, logits = self.compute_loss_and_logits(inputs, labels)
            eval_loss += loss.item()

            predicted = torch.argmax(logits, dim=-1)
            correct += predicted.eq(labels).sum().item()

            if aggregate_epochs and metadata is not None:
                self.update_epoch_aggregation(
                    logit_sums, logit_counts, label_map, logits, labels, metadata
                )
            else:
                y_true = np.concatenate([y_true, labels.reshape(-1).cpu().numpy()])
                y_pred = np.concatenate(
                    [y_pred, logits.reshape(-1, logits.size(-1)).cpu().numpy()]
                )

            progress_bar(
                i,
                len(self.loader_dict[mode]),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (eval_loss / (i + 1), 100.0 * correct / total, correct, total),
            )

        avg_eval_loss = (
            eval_loss / len(self.loader_dict[mode])
            if len(self.loader_dict[mode]) > 0
            else 0
        )

        if aggregate_epochs and logit_sums:
            y_true, y_pred = self.finalize_epoch_aggregation(
                logit_sums, logit_counts, label_map
            )
            correct = (np.argmax(y_pred, axis=1) == y_true).sum()
            total = len(y_true)
            avg_eval_loss = F.cross_entropy(
                torch.from_numpy(y_pred).float(), torch.from_numpy(y_true).long()
            ).item()
            print(
                "\n[INFO] Eval aggregation: mean_logits, unique epochs: {}".format(
                    total
                )
            )

        avg_eval_acc = 100.0 * correct / total if total > 0 else 0.0

        if mode == "val":
            return avg_eval_acc, avg_eval_loss
        if mode == "test":
            return y_true, y_pred
        raise NotImplementedError

    @staticmethod
    def update_epoch_aggregation(
        logit_sums, logit_counts, label_map, logits, labels, metadata
    ):
        logits_np = logits.detach().float().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        file_idx_np = metadata["file_idx"].cpu().numpy()
        epoch_idx_np = metadata["epoch_idx"].cpu().numpy()

        flat_logits = logits_np.reshape(-1, logits_np.shape[-1])
        flat_labels = labels_np.reshape(-1)
        flat_file_idx = file_idx_np.reshape(-1)
        flat_epoch_idx = epoch_idx_np.reshape(-1)

        for file_idx, epoch_idx, logit, label in zip(
            flat_file_idx, flat_epoch_idx, flat_logits, flat_labels
        ):
            key = (int(file_idx), int(epoch_idx))
            if key not in logit_sums:
                logit_sums[key] = logit.copy()
                logit_counts[key] = 1
                label_map[key] = int(label)
            else:
                logit_sums[key] += logit
                logit_counts[key] += 1

    @staticmethod
    def finalize_epoch_aggregation(logit_sums, logit_counts, label_map):
        keys = sorted(logit_sums.keys())
        y_pred = np.stack([logit_sums[key] / logit_counts[key] for key in keys])
        y_true = np.array([label_map[key] for key in keys], dtype=np.int64)
        return y_true, y_pred

    def run(self):
        fold_start_time = time.time()

        for epoch in range(self.tp_cfg["max_epochs"]):
            print("\n[INFO] Fold: {}, Epoch: {}".format(self.fold, epoch))
            self.train_one_epoch(epoch)
            if self.early_stopping.early_stop:
                break

        training_time = time.time() - fold_start_time
        loss_data = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "training_time": training_time,
        }
        os.makedirs(self.ckpt_path, exist_ok=True)
        np.save(
            os.path.join(self.ckpt_path, "losses_fold-{0:02d}.npy".format(self.fold)),
            loss_data,
        )

        self.model.load_state_dict(
            torch.load(os.path.join(self.ckpt_path, self.ckpt_name), map_location=self.device)
        )
        y_true, y_pred = self.evaluate(mode="test")
        print("")
        return y_true, y_pred


def determine_folds_to_train(config, fold):
    if fold is not None:
        return [fold]

    ckpt_path = os.path.join("checkpoints", config["name"])
    if not os.path.exists(ckpt_path):
        print("[INFO] Checkpoint directory does not exist, training all folds")
        return list(range(1, config["dataset"]["num_splits"] + 1))

    folds_to_train = []
    for fold_idx in range(1, config["dataset"]["num_splits"] + 1):
        loss_file = os.path.join(ckpt_path, "losses_fold-{0:02d}.npy".format(fold_idx))
        if not os.path.exists(loss_file):
            folds_to_train.append(fold_idx)
            print("[INFO] Loss file for fold {} not found".format(fold_idx))
        else:
            print("[INFO] Loss file for fold {} found, skipping".format(fold_idx))

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
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument("--fold", type=int, default=None, help="run one fold only")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor")
    parser.add_argument("--amp", action="store_true", help="use CUDA AMP mixed precision")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="enable cudnn benchmark for faster fixed-shape training",
    )
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_random_seed(args.seed, use_cuda=True)
    if args.benchmark and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    with open(args.config) as config_file:
        config = json.load(config_file)
    config["name"] = os.path.basename(args.config).replace(".json", "")
    config["_config_dir"] = os.path.dirname(os.path.abspath(args.config))

    y_true_all = np.zeros(0)
    y_pred_all = np.zeros((0, config["classifier"]["num_classes"]))

    for fold in determine_folds_to_train(config, args.fold):
        trainer = OneFoldSeq2SeqTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        y_true_all = np.concatenate([y_true_all, y_true])
        y_pred_all = np.concatenate([y_pred_all, y_pred])
        summarize_result(config, fold, y_true_all, y_pred_all)


if __name__ == "__main__":
    main()
