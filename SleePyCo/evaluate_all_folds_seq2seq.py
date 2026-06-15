import argparse
import json
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import progress_bar, set_random_seed, summarize_result
from train_seq2seq import Seq2SeqEEGDataLoader, Seq2SeqModel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def autocast_cuda(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.autocast(enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def normalize_state_dict_for_model(model, state_dict):
    model_is_parallel = hasattr(model, "module")
    state_is_parallel = any(key.startswith("module.") for key in state_dict)

    if model_is_parallel and not state_is_parallel:
        return {"module." + key: value for key, value in state_dict.items()}

    if not model_is_parallel and state_is_parallel:
        return {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

    return state_dict

def parse_folds(folds_arg, num_splits):
    if folds_arg is None:
        return list(range(1, num_splits + 1))

    folds = []
    for part in folds_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            folds.extend(range(int(start), int(end) + 1))
        else:
            folds.append(int(part))

    return sorted(dict.fromkeys(folds))


def build_loader(config, fold, args, device):
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else min(16, max(4, 2 * len(args.gpu.split(","))))
    )
    dataset = Seq2SeqEEGDataLoader(
        config, fold, set="test", return_metadata=True
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size or config["training_params"]["batch_size"],
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**loader_kwargs)


def build_model(config, args, device):
    model = Seq2SeqModel(config)
    gpu_ids = [gpu_id for gpu_id in args.gpu.split(",") if gpu_id != ""]
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(len(gpu_ids))))
    model.to(device)
    model.eval()
    return model


def resolve_checkpoint(config, fold, ckpt_dir):
    base_dir = ckpt_dir or os.path.join(SCRIPT_DIR, "checkpoints", config["name"])
    ckpt_path = os.path.join(base_dir, "ckpt_fold-{0:02d}.pth".format(fold))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    return ckpt_path


def load_checkpoint(model, ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = normalize_state_dict_for_model(model, state_dict)
    model.load_state_dict(state_dict, strict=False)


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


def finalize_epoch_aggregation(logit_sums, logit_counts, label_map):
    keys = sorted(logit_sums.keys())
    y_pred = np.stack([logit_sums[key] / logit_counts[key] for key in keys])
    y_true = np.array([label_map[key] for key in keys], dtype=np.int64)
    return y_true, y_pred


@torch.no_grad()
def evaluate_fold(model, loader, config, args, device):
    model.eval()
    use_amp = bool(args.amp and device.type == "cuda")
    eval_strategy = args.eval_strategy or config["training_params"].get(
        "eval_strategy", "mean_logits"
    )
    aggregate_epochs = eval_strategy == "mean_logits"

    eval_loss = 0
    correct = 0
    total = 0
    y_true = np.zeros(0)
    y_pred = np.zeros((0, config["classifier"]["num_classes"]))
    logit_sums, logit_counts, label_map = {}, {}, {}

    for i, (inputs, labels, metadata) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        flat_labels = labels.reshape(-1)

        with autocast_cuda(use_amp):
            outputs = model(inputs)
            logits = torch.zeros_like(outputs[0])
            loss = 0
            for output in outputs:
                loss += F.cross_entropy(
                    output.reshape(-1, output.size(-1)), flat_labels
                )
                logits += output

        eval_loss += loss.item()
        predicted = torch.argmax(logits, dim=-1)
        correct += predicted.eq(labels).sum().item()
        total += labels.numel()

        if aggregate_epochs:
            update_epoch_aggregation(
                logit_sums, logit_counts, label_map, logits, labels, metadata
            )
        else:
            y_true = np.concatenate([y_true, labels.reshape(-1).cpu().numpy()])
            y_pred = np.concatenate(
                [y_pred, logits.reshape(-1, logits.size(-1)).cpu().numpy()]
            )

        progress_bar(
            i,
            len(loader),
            "Loss: %.3f | Window Acc: %.3f%% (%d/%d)"
            % (eval_loss / (i + 1), 100.0 * correct / total, correct, total),
        )

    if aggregate_epochs:
        y_true, y_pred = finalize_epoch_aggregation(
            logit_sums, logit_counts, label_map
        )
        print("[INFO] Eval aggregation: mean_logits, unique epochs: {}".format(len(y_true)))

    return y_true, y_pred


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="fold list, for example '1,2,3' or '1-10'",
    )
    parser.add_argument("--ckpt-dir", type=str, default=None, help="checkpoint dir")
    parser.add_argument("--batch-size", type=int, default=None, help="evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor")
    parser.add_argument(
        "--eval-strategy",
        choices=["mean_logits", "window_flatten"],
        default=None,
        help="how to handle repeated sliding-window predictions",
    )
    parser.add_argument("--amp", action="store_true", help="use CUDA AMP mixed precision")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="enable cudnn benchmark for faster fixed-shape evaluation",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = parse_folds(args.folds, config["dataset"]["num_splits"])
    print("[INFO] Evaluating folds: {}".format(folds))

    y_true_all = np.zeros(0)
    y_pred_all = np.zeros((0, config["classifier"]["num_classes"]))

    for fold in folds:
        print("\n[INFO] Evaluating fold {}".format(fold))
        loader = build_loader(config, fold, args, device)
        model = build_model(config, args, device)
        ckpt_path = resolve_checkpoint(config, fold, args.ckpt_dir)
        print("[INFO] Loading checkpoint: {}".format(ckpt_path))
        load_checkpoint(model, ckpt_path, device)

        y_true, y_pred = evaluate_fold(model, loader, config, args, device)
        print("")
        summarize_result(config, fold, y_true, y_pred, save=False)

        y_true_all = np.concatenate([y_true_all, y_true])
        y_pred_all = np.concatenate([y_pred_all, y_pred])

    print("\n[INFO] Pooled result across folds: {}".format(folds))
    summarize_result(config, "all", y_true_all, y_pred_all)


if __name__ == "__main__":
    main()
