"""Unified training launcher for the sleep project.

This file keeps the existing training scripts intact and provides one
predictable command surface for day-to-day runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent
PATH_ARGUMENTS = {"--config", "--pretrain_checkpoint"}


@dataclass(frozen=True)
class TrainCommand:
    project_dir: str
    script: str
    description: str


COMMANDS: Dict[str, Dict[str, TrainCommand]] = {
    "sleepyco": {
        "crl": TrainCommand(
            "SleePyCo",
            "train_crl.py",
            "Supervised contrastive representation pretraining.",
        ),
        "crl-frea": TrainCommand(
            "SleePyCo",
            "train_crl_FreRA.py",
            "Contrastive pretraining with FreRA frequency augmentation.",
        ),
        "crl-mix": TrainCommand(
            "SleePyCo",
            "train_crl_mix.py",
            "Contrastive pretraining with mixed time-domain and FreRA views.",
        ),
        "no-label": TrainCommand(
            "SleePyCo",
            "train_no_label.py",
            "Self-supervised contrastive pretraining without labels.",
        ),
        "mtcl": TrainCommand(
            "SleePyCo",
            "train_mtcl.py",
            "Multi-scale temporal context learning / standard fine-tuning.",
        ),
        "seq2seq": TrainCommand(
            "SleePyCo",
            "train_seq2seq.py",
            "Sequence-to-sequence fine-tuning over all epochs in a window.",
        ),
        "scratch": TrainCommand(
            "SleePyCo",
            "train_scratch.py",
            "Supervised training from scratch.",
        ),
        "mtcl-fullfinetune": TrainCommand(
            "SleePyCo",
            "train_mtcl_fullfinetune.py",
            "MTCL full fine-tuning variant.",
        ),
        "mtcl-fullfinetune-2048": TrainCommand(
            "SleePyCo",
            "train_mtcl_fullfinetune2048.py",
            "MTCL full fine-tuning variant with the 2048 setup.",
        ),
        "mae": TrainCommand(
            "SleePyCo",
            "train_mae.py",
            "Masked autoencoder pretraining.",
        ),
        "finetune-mae": TrainCommand(
            "SleePyCo",
            "train_finetuneMAE.py",
            "Fine-tune a pretrained MAE model.",
        ),
        "fullfinetune-mae": TrainCommand(
            "SleePyCo",
            "train_fullfinetuneMAE.py",
            "Full fine-tune a pretrained MAE model.",
        ),
    },
    "jepa": {
        "pretrain": TrainCommand(
            "sleepJEPA",
            "train_pretrain.py",
            "JEPA self-supervised pretraining.",
        ),
        "finetune": TrainCommand(
            "sleepJEPA",
            "train_finetune.py",
            "JEPA downstream fine-tuning.",
        ),
    },
}


ALIASES: Dict[str, Dict[str, str]] = {
    "sleepyco": {
        "contrastive": "crl",
        "frea": "crl-frea",
        "mix-frea": "crl-mix",
        "pretrain": "crl",
        "finetune": "mtcl",
        "freezefinetune": "mtcl",
        "sequence": "seq2seq",
        "sequence2sequence": "seq2seq",
        "fullfinetune": "mtcl-fullfinetune",
        "mae-pretrain": "mae",
    },
    "jepa": {},
}


def iter_command_lines() -> Iterable[str]:
    for project in sorted(COMMANDS):
        yield f"{project}:"
        for task, command in sorted(COMMANDS[project].items()):
            yield f"  {task:<24} {command.project_dir}/{command.script:<32} {command.description}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run project training scripts through a single launcher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available tasks:\n" + "\n".join(iter_command_lines()),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print registered training tasks and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the legacy command without running it",
    )
    parser.add_argument(
        "project",
        nargs="?",
        choices=sorted(COMMANDS),
        help="training project",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="registered task name",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="arguments passed through to the legacy training script",
    )
    return parser


def normalize_task(project: str, task: str) -> str:
    alias = ALIASES.get(project, {}).get(task)
    return alias or task


def strip_separator(args: List[str]) -> List[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def pop_launcher_flags(args: List[str]) -> Tuple[List[str], bool]:
    script_args: List[str] = []
    dry_run = False

    for arg in args:
        if arg == "--dry-run":
            dry_run = True
            continue
        script_args.append(arg)

    return script_args, dry_run


def resolve_path_argument(value: str, project_path: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return value

    root_candidate = ROOT / path
    if root_candidate.exists():
        return str(root_candidate)

    project_candidate = project_path / path
    if project_candidate.exists():
        return str(project_candidate)

    return value


def normalize_script_args(args: List[str], project_path: Path) -> List[str]:
    normalized: List[str] = []
    index = 0

    while index < len(args):
        arg = args[index]

        if arg in PATH_ARGUMENTS and index + 1 < len(args):
            normalized.append(arg)
            normalized.append(resolve_path_argument(args[index + 1], project_path))
            index += 2
            continue

        matched_path_arg = False
        for path_arg in PATH_ARGUMENTS:
            prefix = path_arg + "="
            if arg.startswith(prefix):
                normalized.append(prefix + resolve_path_argument(arg[len(prefix) :], project_path))
                matched_path_arg = True
                break

        if not matched_path_arg:
            normalized.append(arg)

        index += 1

    return normalized


def resolve_command(project: str, task: str) -> TrainCommand:
    normalized_task = normalize_task(project, task)
    try:
        return COMMANDS[project][normalized_task]
    except KeyError as exc:
        known = ", ".join(sorted(COMMANDS[project]))
        aliases = ", ".join(sorted(ALIASES.get(project, {})))
        detail = f"Known tasks: {known}"
        if aliases:
            detail += f"\nAliases: {aliases}"
        raise SystemExit(f"Unknown task '{task}' for project '{project}'.\n{detail}") from exc


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print("\n".join(iter_command_lines()))
        return 0

    if not args.project or not args.task:
        parser.print_help()
        return 2

    command = resolve_command(args.project, args.task)
    script_args = strip_separator(args.script_args)
    script_args, trailing_dry_run = pop_launcher_flags(script_args)
    project_path = ROOT / command.project_dir
    script_path = project_path / command.script

    if not script_path.exists():
        raise SystemExit(f"Training script not found: {script_path}")

    script_args = normalize_script_args(script_args, project_path)
    legacy_command = [sys.executable, command.script, *script_args]
    print(f"Project: {command.project_dir}")
    print(f"Task: {args.task}")
    print("Command: " + " ".join(legacy_command))

    if args.dry_run or trailing_dry_run:
        return 0

    return subprocess.call(legacy_command, cwd=str(project_path))


if __name__ == "__main__":
    raise SystemExit(main())
