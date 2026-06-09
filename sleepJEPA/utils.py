import os
import random
import numpy as np
import torch


def set_random_seed(seed: int = 42, use_cuda: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def progress_bar(current: int, total: int, msg: str = None, bar_len: int = 30):
    if total == 0:
        return
    progress = float(current + 1) / total
    filled_len = int(bar_len * progress)
    bar = "#" * filled_len + "-" * (bar_len - filled_len)
    if msg is None:
        msg = ""
    print(f"[{bar}] {current + 1}/{total} {msg}", end="\r")
    if current + 1 == total:
        print()


def make_dirs(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state: dict, path: str):
    make_dirs(os.path.dirname(path))
    torch.save(state, path)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0.0,
        ckpt_path: str = "./checkpoints",
        ckpt_name: str = "checkpoint.pth",
        mode: str = "min",
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode
        if mode == "max":
            self.init_metric = -np.inf
        elif mode == "min":
            self.init_metric = np.inf
        else:
            raise NotImplementedError(f"Unsupported mode: {mode}")

        self.delta = delta
        self.ckpt_path = ckpt_path
        self.ckpt_name = ckpt_name if ckpt_name.endswith(".pth") else ckpt_name + ".pth"
        os.makedirs(self.ckpt_path, exist_ok=True)

    def __call__(
        self,
        val_acc: float | None,
        val_loss: float | None,
        model: torch.nn.Module,
    ):
        if self.mode == "max":
            if val_acc is None:
                raise ValueError("val_acc must be provided for mode='max'.")
            score = val_acc
            val_metric = val_acc
        elif self.mode == "min":
            if val_loss is None:
                raise ValueError("val_loss must be provided for mode='min'.")
            score = -val_loss
            val_metric = val_loss
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric: float, model: torch.nn.Module):
        if self.verbose:
            if self.mode == "max":
                print(
                    f"[INFO] Validation accuracy increased ({self.init_metric:.6f} --> {val_metric:.6f}). Saving model ...\n"
                )
            else:
                print(
                    f"[INFO] Validation loss decreased ({self.init_metric:.6f} --> {val_metric:.6f}). Saving model ...\n"
                )
        torch.save(model.state_dict(), os.path.join(self.ckpt_path, self.ckpt_name))
        self.init_metric = val_metric
