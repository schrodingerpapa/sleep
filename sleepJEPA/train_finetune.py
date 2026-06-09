import argparse
import json
import os
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from loader import EEGJEPADataLoader
from models.jepa_encoder import JEPAModel
from models import JEPA
from utils import (
    EarlyStopping,
    make_dirs,
    progress_bar,
    save_checkpoint,
    set_random_seed,
)


class JEPAFinetuneModel(nn.Module):
    def __init__(
        self, encoder: JEPAModel, num_classes: int = 5, freeze_encoder: bool = False
    ):
        super().__init__()
        self.encoder = encoder.ctx_encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feature = self.encoder(x)
        pooled = self.pool(feature.transpose(1, 2)).squeeze(-1)
        return self.classifier(pooled)


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    val_loader=None,
    early_stopper=None,
    total_iter=0,
    val_period=1000,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)
        total_iter += 1

        if (
            val_loader is not None
            and early_stopper is not None
            and total_iter % val_period == 0
        ):
            val_metrics = evaluate(model, val_loader, criterion, device, return_all_metrics=False)
            val_loss, val_acc = val_metrics[0], val_metrics[1]
            print(
                f"[INFO] Iter {total_iter}: val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
            )
            early_stopper(val_acc, val_loss, model)
            model.train()
            if early_stopper.early_stop:
                return (
                    total_loss / max(batch_idx + 1, 1),
                    100.0 * total_correct / max(total_samples, 1),
                    total_iter,
                    True,
                )

        progress_bar(
            batch_idx,
            len(loader),
            f"Loss: {total_loss / (batch_idx + 1):.4f} Acc: {100.0 * total_correct / total_samples:.2f}%",
        )

    return (
        total_loss / max(len(loader), 1),
        100.0 * total_correct / max(total_samples, 1),
        total_iter,
        False,
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_all_metrics=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true = np.zeros(0)
    y_pred = np.zeros(0)

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)
        
        # 收集所有真实标签和预测标签用于计算更多指标
        y_true = np.concatenate([y_true, labels.cpu().numpy()])
        y_pred = np.concatenate([y_pred, predicted.cpu().numpy()])

        progress_bar(
            batch_idx,
            len(loader),
            f"Eval Loss: {total_loss / (batch_idx + 1):.4f} Acc: {100.0 * total_correct / total_samples:.2f}%",
        )
    
    accuracy = 100.0 * total_correct / max(total_samples, 1)
    
    if not return_all_metrics:
        return total_loss / max(len(loader), 1), accuracy
    else:
        # 计算所有需要的指标：accuracy, macro_f1, kappa, wf1, n1f1, n2f1, n3f1, rf1
        # 假设睡眠阶段标签：0=W, 1=N1, 2=N2, 3=N3, 4=R（根据常见的睡眠阶段划分）
        macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
        wf1 = f1_score(y_true, y_pred, average='weighted') * 100
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # 计算每个类别的F1
        class_f1 = f1_score(y_true, y_pred, average=None)
        wf1_perclass = class_f1[0] * 100 if len(class_f1) > 0 else 0.0
        n1f1 = class_f1[1] * 100 if len(class_f1) > 1 else 0.0
        n2f1 = class_f1[2] * 100 if len(class_f1) > 2 else 0.0
        n3f1 = class_f1[3] * 100 if len(class_f1) > 3 else 0.0
        rf1 = class_f1[4] * 100 if len(class_f1) > 4 else 0.0
        
        metrics = {
            'loss': total_loss / max(len(loader), 1),
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'kappa': kappa,
            'wf1': wf1,
            'wf1_perclass': wf1_perclass,
            'n1f1': n1f1,
            'n2f1': n2f1,
            'n3f1': n3f1,
            'rf1': rf1
        }
        return metrics


def load_pretrained_encoder(
    checkpoint_path: str, model: JEPAModel, device: torch.device
):
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state
    ctx_state = {k: v for k, v in state_dict.items() if k.startswith("ctx_encoder.")}
    model.load_state_dict(ctx_state, strict=False)


def run_finetune(config, args, fold):
    """针对单个fold运行微调流程，支持DDP分布式多GPU训练"""
    # 初始化DDP
    rank = 0
    local_rank = 0
    world_size = 1
    is_distributed = False
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = True
        if local_rank == 0:
            print(f"\n{'='*60}")
            print(f"[INFO] Starting fine-tuning for fold {fold:02d} (DDP, world_size={world_size})")
            print(f"{'='*60}")
    else:
        if args.gpu:
            print(f"\n{'='*60}")
            print(f"[INFO] Starting fine-tuning for fold {fold:02d} (single GPU)")
            print(f"{'='*60}")
        # 为每个fold设置不同的随机种子，避免种子重复
        set_random_seed(args.seed + fold, use_cuda=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if local_rank == 0:
            print(f"[INFO] Using device: {device}")

    # 加载当前fold的数据集
    train_dataset = EEGJEPADataLoader(config, fold=fold, set="train")
    val_dataset = EEGJEPADataLoader(config, fold=fold, set="val")
    test_dataset = EEGJEPADataLoader(config, fold=fold, set="test")
    if local_rank == 0:
        print(f"[INFO] fold{fold:02d} - Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # DDP使用DistributedSampler分配数据到各个GPU
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training_params"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["training_params"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training_params"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["training_params"].get("num_workers", 4),
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training_params"]["batch_size"],
        shuffle=False,
        sampler=test_sampler,
        num_workers=config["training_params"].get("num_workers", 4),
        pin_memory=True,
    )

    # 初始化模型
    jepa_model = JEPA(
        patch_len=config["model"]["patch_len"],
        seq_len=config["model"]["seq_len"],
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        ctx_len=config["model"]["ctx_len"],
        tgt_len=config["model"]["tgt_len"],
    ).to(device)

    # 决定当前fold对应的预训练权重位置：支持传入文件（精确到pth）或父目录（包含 fold_XX 子文件夹）
    fold_str = f"fold_{fold:02d}"
    # 优先使用命令行参数，其次使用配置文件里的路径，如果都没有则使用原来的默认目录
    pretrain_arg = args.pretrain_checkpoint if args.pretrain_checkpoint else config["training_params"].get("pretrain_checkpoint", "")
    if not pretrain_arg:
        pretrain_arg = "/home/chenlungan/算法模型/sleepJEPA/checkpoints/JEPA-Transformer_SL-01_Sleep-EDF-2018_pretrain"

    pretrain_checkpoint_path = None
    # 如果提供的是目录，则在其下查找对应fold的best_ckpt.pth
    if os.path.isdir(pretrain_arg):
        candidate = os.path.join(pretrain_arg, fold_str, "best_ckpt.pth")
        if os.path.exists(candidate):
            pretrain_checkpoint_path = candidate
    # 如果提供的是文件路径且存在，直接使用
    elif os.path.isfile(pretrain_arg):
        pretrain_checkpoint_path = pretrain_arg
    else:
        # 兜底：尝试把提供的路径当作基路径拼接子文件夹
        candidate = os.path.join(pretrain_arg, fold_str, "best_ckpt.pth")
        if os.path.exists(candidate):
            pretrain_checkpoint_path = candidate

    if pretrain_checkpoint_path and os.path.exists(pretrain_checkpoint_path):
        load_pretrained_encoder(pretrain_checkpoint_path, jepa_model, device)
        print(f"[INFO] Loaded pretrained weights from: {pretrain_checkpoint_path}")
    else:
        print(f"[WARNING] Pretrained checkpoint not found at {pretrain_arg} (or expected fold subfolder). Training from scratch.")

    finetune_model = JEPAFinetuneModel(
        jepa_model,
        num_classes=config["classifier"]["num_classes"],
        freeze_encoder=config["training_params"].get("freeze_encoder", False),
    ).to(device)
    
    # DDP模式下用DistributedDataParallel包装模型
    if is_distributed:
        finetune_model = DDP(finetune_model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(
        [p for p in finetune_model.parameters() if p.requires_grad],
        lr=config["training_params"]["lr"],
        weight_decay=config["training_params"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    # 为每个fold创建独立的保存目录，避免结果覆盖
    base_ckpt_dir = os.path.join("checkpoints", config["name"])
    base_result_dir = os.path.join("results", config["name"])
    ckpt_dir = os.path.join(base_ckpt_dir, f"fold_{fold:02d}")
    result_dir = os.path.join(base_result_dir, f"fold_{fold:02d}")
    make_dirs(ckpt_dir)
    make_dirs(result_dir)

    best_val = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    early_stopper = EarlyStopping(
        patience=config["training_params"]["early_stopping"]["patience"],
        verbose=True,
        ckpt_path=ckpt_dir,
        ckpt_name="best_ckpt",
        mode=config["training_params"]["early_stopping"].get("mode", "max"),
        delta=config["training_params"]["early_stopping"].get("delta", 0.0),
    )
    total_iter = 0
    val_period = config["training_params"].get("val_period", 1000)

    for epoch in range(1, config["training_params"]["max_epochs"] + 1):
        print(f"[INFO] Epoch {epoch}/{config['training_params']['max_epochs']} (fold {fold:02d})")
        train_loss, train_acc, total_iter, stopped = train_epoch(
            finetune_model,
            train_loader,
            optimizer,
            criterion,
            device,
            val_loader=val_loader,
            early_stopper=early_stopper,
            total_iter=total_iter,
            val_period=val_period,
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_metrics = evaluate(finetune_model, val_loader, criterion, device, return_all_metrics=False)
        val_loss, val_acc = val_metrics[0], val_metrics[1]
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[INFO] fold{fold:02d} train_loss={train_loss:.4f} train_acc={train_acc:.2f}% val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val and local_rank == 0:
            best_val = val_acc
            # 保存模型只在主进程执行，避免多进程写入冲突
            if is_distributed:
                model_state = finetune_model.module.state_dict()
            else:
                model_state = finetune_model.state_dict()
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                os.path.join(ckpt_dir, f"best_ckpt_epoch_{epoch:03d}.pth"),
            )

        if stopped and local_rank == 0:
            print(f"[INFO] Early stopping triggered at iteration {total_iter} for fold {fold:02d}.")
            break
    
    # 等待所有进程同步完成
    if is_distributed:
        dist.barrier()

    # 测试集评估，返回所有指标（只在主进程获取完整指标）
    test_metrics = evaluate(finetune_model, test_loader, criterion, device, return_all_metrics=True)
    test_loss = test_metrics['loss']
    test_acc = test_metrics['accuracy']
    
    if local_rank == 0:
        print(f"[INFO] fold{fold:02d} Final test loss={test_loss:.4f} test acc={test_acc:.2f}%")
        save_checkpoint(history, os.path.join(result_dir, "history.pth"))
        
        print(f"[INFO] Completed fine-tuning for fold {fold:02d}")
    
    # 销毁DDP进程组，准备下一个fold
    if is_distributed:
        dist.destroy_process_group()
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="JEPA fine-tuning for EEG downstream task - auto process all folds"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/jepa_finetune_sleepedf.json",
        help="config file path",
    )
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default="/home/chenlungan/算法模型/sleepJEPA/checkpoints/JEPA-Transformer_SL-01_Sleep-EDF-2018_pretrain",
        help="pretrained JEPA checkpoint path",
    )
    parser.add_argument(
        "--start_fold", type=int, default=1, help="start from this fold index"
    )
    parser.add_argument(
        "--end_fold", type=int, default=10, help="end at this fold index"
    )
    parser.add_argument("--gpu", type=str, default="0,1,2,3,4,5,6,7", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"[INFO] Using CUDA_VISIBLE_DEVICES: {args.gpu}")
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["name"] = config.get(
        "name", os.path.splitext(os.path.basename(args.config))[0]
    )
    print(f"[INFO] Config name: {config['name']}")

    # 创建汇总结果目录
    base_result_dir = os.path.join("results", config["name"])
    make_dirs(base_result_dir)
    summary_path = os.path.join(base_result_dir, "all_folds_results.txt")
    # 初始化汇总文件，写入表头，和SleePyCo格式一致
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("fold accuracy macro_f1 kappa wf1 n1f1 n2f1 n3f1 rf1\n")

    # 自动循环处理所有fold - 这就是自动处理10个fold的核心逻辑
    all_test_metrics = []
    for fold in range(args.start_fold, args.end_fold + 1):
        test_metrics = run_finetune(config, args, fold)
        all_test_metrics.append(test_metrics)
        
        # 立即将当前fold的结果写入汇总文件
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"fold{fold:02d} {test_metrics['accuracy']:.4f} {test_metrics['macro_f1']:.4f} {test_metrics['kappa']:.4f} "
                    f"{test_metrics['wf1']:.4f} {test_metrics['n1f1']:.4f} {test_metrics['n2f1']:.4f} "
                    f"{test_metrics['n3f1']:.4f} {test_metrics['rf1']:.4f}\n")
        
        # 清理GPU缓存，为下一个fold准备
        torch.cuda.empty_cache()
        print(f"[INFO] GPU cache cleared for next fold")

    # 计算平均指标
    avg_metrics = {}
    for key in all_test_metrics[0].keys():
        if key not in ['loss']:  # 只平均计算指标
            avg_metrics[key] = sum(m[key] for m in all_test_metrics) / len(all_test_metrics)
    
    # 打印所有fold的汇总结果
    print(f"\n{'='*80}")
    print("[INFO] All folds completed! Summary results:")
    print(f"{'='*80}")
    print(f"{'fold':<8} {'accuracy':<10} {'macro_f1':<10} {'kappa':<8} {'wf1':<8} {'n1f1':<8} {'n2f1':<8} {'n3f1':<8} {'rf1':<8}")
    print(f"{'-'*80}")
    
    # 打印每个fold的结果
    for i, metrics in enumerate(all_test_metrics):
        fold = args.start_fold + i
        print(f"fold{fold:02d} {metrics['accuracy']:.4f}    {metrics['macro_f1']:.4f}    {metrics['kappa']:.4f}  {metrics['wf1']:.4f}  {metrics['n1f1']:.4f}  {metrics['n2f1']:.4f}  {metrics['n3f1']:.4f}  {metrics['rf1']:.4f}")
    
    print(f"{'-'*80}")
    print(f"Average {avg_metrics['accuracy']:.4f}    {avg_metrics['macro_f1']:.4f}    {avg_metrics['kappa']:.4f}  {avg_metrics['wf1']:.4f}  {avg_metrics['n1f1']:.4f}  {avg_metrics['n2f1']:.4f}  {avg_metrics['n3f1']:.4f}  {avg_metrics['rf1']:.4f}")
    print(f"{'='*80}")
    
    # 保存平均结果到配置目录下的汇总文件
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"\nAverage {avg_metrics['accuracy']:.4f} {avg_metrics['macro_f1']:.4f} {avg_metrics['kappa']:.4f} "
                f"{avg_metrics['wf1']:.4f} {avg_metrics['n1f1']:.4f} {avg_metrics['n2f1']:.4f} "
                f"{avg_metrics['n3f1']:.4f} {avg_metrics['rf1']:.4f}\n")
    
    # 在你要求的路径生成全局汇总文件：/home/chenlungan/算法模型/sleepJEPA/results/下，和SleePyCo格式一致
    global_summary_path = "/home/chenlungan/算法模型/sleepJEPA/results/JEPA-Transformer_finetune_results.txt"
    # 如果文件不存在，先写入表头
    if not os.path.exists(global_summary_path):
        with open(global_summary_path, "w", encoding="utf-8") as f:
            f.write("fold accuracy macro_f1 kappa wf1 n1f1 n2f1 n3f1 rf1\n")
    
    # 追加当前配置的所有fold结果和平均结果
    with open(global_summary_path, "a", encoding="utf-8") as f:
        f.write(f"\n# Results for {config['name']}, processed folds: {args.start_fold}-{args.end_fold}\n")
        # 写入所有fold的结果
        with open(summary_path, "r", encoding="utf-8") as f2:
            lines = f2.readlines()[1:-1]  # 跳过表头和最后一行的Average
            f.writelines(lines)
        # 写入平均结果
        f.write(f"Average {avg_metrics['accuracy']:.4f} {avg_metrics['macro_f1']:.4f} {avg_metrics['kappa']:.4f} "
                f"{avg_metrics['wf1']:.4f} {avg_metrics['n1f1']:.4f} {avg_metrics['n2f1']:.4f} "
                f"{avg_metrics['n3f1']:.4f} {avg_metrics['rf1']:.4f}\n")
    
    print(f"[INFO] Global summary saved to: {global_summary_path}")


if __name__ == "__main__":
    main()