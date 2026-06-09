import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from loader import EEGJEPADataLoader
from models.jepa_encoder import JEPAModel
from utils import (
    EarlyStopping,
    make_dirs,
    progress_bar,
    save_checkpoint,
    set_random_seed,
)


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    config,
    val_loader=None,
    early_stopper=None,
    total_iter=0,
):
    model.train()
    total_loss = 0.0
    val_period = config["training_params"].get("val_period", 1000)
    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)
        else:
            inputs = batch.to(device)

        optimizer.zero_grad()
        pred_feat, tgt_feat = model(inputs)
        loss = criterion(pred_feat, tgt_feat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iter += 1

        if (
            val_loader is not None
            and early_stopper is not None
            and total_iter % val_period == 0
        ):
            val_loss = evaluate(model, val_loader, criterion, device, config)
            print(f"[INFO] Iter {total_iter}: validation loss = {val_loss:.4f}")
            early_stopper(None, val_loss, model)
            if early_stopper.early_stop:
                return total_loss / max(batch_idx + 1, 1), total_iter, True

        progress_bar(
            batch_idx, len(loader), f"Loss: {total_loss / (batch_idx + 1):.4f}"
        )

    return total_loss / max(len(loader), 1), total_iter, False


def evaluate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)

            pred_feat, tgt_feat = model(inputs)
            loss = criterion(pred_feat, tgt_feat)
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="JEPA pretraining for EEG transformer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/jepa_pretrain_sleepedf.json",
        help="config file path",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="fold index for dataset split; 0 means train all folds sequentially",
    )
    parser.add_argument(
        "--gpu", type=str, default="0,1,2,3,4,5,6,7", help="CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["name"] = config.get(
        "name", os.path.splitext(os.path.basename(args.config))[0]
    )

    set_random_seed(args.seed, use_cuda=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [int(x) for x in args.gpu.split(",") if x.strip()]

    folds = (
        list(range(1, config["dataset"]["num_splits"] + 1))
        if args.fold == 0
        else [args.fold]
    )

    for fold in folds:
        print(f"\n[INFO] Starting fold {fold}/{len(folds)}")

        train_dataset = EEGJEPADataLoader(config, fold=fold, set="train")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config["training_params"]["batch_size"],
            shuffle=True,
            num_workers=config["training_params"].get("num_workers", 4),
            pin_memory=True,
        )

        val_loader = None
        if config["training_params"].get("use_val", False):
            val_dataset = EEGJEPADataLoader(config, fold=fold, set="val")
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=config["training_params"]["batch_size"],
                shuffle=False,
                num_workers=config["training_params"].get("num_workers", 4),
                pin_memory=True,
            )

        model = JEPAModel(
            patch_len=config["model"]["patch_len"],
            seq_len=config["model"]["seq_len"],
            embed_dim=config["model"]["embed_dim"],
            num_heads=config["model"]["num_heads"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            ctx_len=config["model"]["ctx_len"],
            tgt_len=config["model"]["tgt_len"],
        ).to(device)

        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["training_params"]["lr"],
            weight_decay=config["training_params"]["weight_decay"],
        )
        criterion = nn.MSELoss()

        ckpt_dir = os.path.join("checkpoints", config["name"], f"fold_{fold:02d}")
        result_dir = os.path.join("results", config["name"], f"fold_{fold:02d}")
        make_dirs(ckpt_dir)
        make_dirs(result_dir)

        history = {"train_loss": [], "val_loss": []}
        early_stopper = None
        if val_loader is not None:
            early_stopper = EarlyStopping(
                patience=config["training_params"]["early_stopping"]["patience"],
                verbose=True,
                ckpt_path=ckpt_dir,
                ckpt_name="best_ckpt",
                mode=config["training_params"]["early_stopping"].get("mode", "min"),
            )

        total_iter = 0
        for epoch in range(1, config["training_params"]["max_epochs"] + 1):
            print(
                f"[INFO] Fold {fold} Epoch {epoch}/{config['training_params']['max_epochs']}"
            )
            train_loss, total_iter, stopped = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                config,
                val_loader=val_loader,
                early_stopper=early_stopper,
                total_iter=total_iter,
            )
            history["train_loss"].append(train_loss)

            if stopped:
                print(
                    f"[INFO] Early stopping triggered at iteration {total_iter} for fold {fold}."
                )
                break

            # Validation is handled inside the training loop at iteration-level
            # following SleePyCo: rely on val_period checks and EarlyStopping.
            print(f"[INFO] train_loss={train_loss:.4f}")

            # 只在最后一轮保存模型，或者由EarlyStopping保存最佳模型
            # EarlyStopping会自动保存验证集上的最佳模型到best_ckpt.pth
            if epoch == config["training_params"]["max_epochs"]:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                    },
                    os.path.join(ckpt_dir, "final_ckpt.pth"),
                )

        save_checkpoint(history, os.path.join(result_dir, "history.pth"))

    if args.fold == 0:
        print(f"\n[INFO] All {len(folds)} folds finished.")


if __name__ == "__main__":
    main()
