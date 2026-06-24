#!/usr/bin/env python
"""Multi-domain EEG visualization and pairwise MMD analysis.

This is a thin wrapper around domain_shift_visualization.py for comparing more
than two datasets, including unlabeled datasets. Unlabeled NPZ files are allowed
as long as they contain x; their stage label is reported as Unknown.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys

import numpy as np

import domain_shift_visualization as dsv


DEFAULT_UNLABELED_DATASETS = {"EEG-Monitor"}

DEFAULT_DATASETS = [
    "DODH=/document/chenlungan/datasets/DODH/npz/Fp1-Fp2",
    "AnphySleep=/document/chenlungan/datasets/AnphySleep/npz/Fp1-Fp2",
    "EEG-Monitor=/document/chenlungan/datasets/EEG-Monitor/npz/Fp1-Fp2",
]


def parse_dataset_spec(spec):
    if "=" not in spec:
        raise ValueError("Dataset spec must be NAME=PATH, got: {}".format(spec))
    name, root = spec.split("=", 1)
    name = name.strip()
    root = root.strip()
    if not name or not root:
        raise ValueError("Dataset spec must be NAME=PATH, got: {}".format(spec))
    return {"name": name, "root": root}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple EEG NPZ datasets with t-SNE and pairwise MMD."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset specification as NAME=PATH. Repeat for three or more datasets.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("outputs", "domain_shift_multidomain_fp1_fp2"),
        help="Directory where plots and statistics will be written.",
    )
    parser.add_argument(
        "--analysis-label",
        default="FP1-FP2",
        help="Short label used in plot titles and summaries, for example FP1-FP2 or mixed FP1-FP2/Fpz-Cz.",
    )
    parser.add_argument(
        "--unlabeled-dataset",
        action="append",
        default=None,
        help=(
            "Dataset name whose labels should be ignored and treated as Unknown. "
            "Repeat for multiple datasets. Defaults to EEG-Monitor."
        ),
    )
    parser.add_argument(
        "--no-default-unlabeled",
        action="store_true",
        help="Do not automatically treat EEG-Monitor as unlabeled.",
    )

    parser.add_argument("--signal-key", default="x", help="NPZ key for epoch signals.")
    parser.add_argument("--label-key", default="y", help="NPZ key for epoch labels.")
    parser.add_argument("--fs-key", default="fs", help="NPZ key for sampling rate.")
    parser.add_argument("--default-fs", type=float, default=100.0)
    parser.add_argument("--keep-not-scored", action="store_true")
    parser.add_argument("--stage-filter", default=None)

    parser.add_argument("--sample-mode", choices=["balanced", "random", "all"], default="balanced")
    parser.add_argument("--max-epochs-per-domain", type=int, default=3000)
    parser.add_argument("--random-seed", type=int, default=2026)

    parser.add_argument("--feature-mode", choices=["summary", "raw"], default="summary")
    parser.add_argument("--raw-points", type=int, default=300)
    parser.add_argument("--per-epoch-zscore", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--no-standardize", action="store_true")

    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true")

    parser.add_argument("--mmd-kernel", choices=["rbf", "linear"], default="rbf")
    parser.add_argument("--mmd-sigmas", default="median")
    parser.add_argument("--mmd-max-per-domain", type=int, default=1000)
    parser.add_argument("--mmd-permutations", type=int, default=200)
    return parser.parse_args()


def unlabeled_dataset_names(args):
    names = set()
    if not args.no_default_unlabeled:
        names.update(DEFAULT_UNLABELED_DATASETS)
    if args.unlabeled_dataset:
        names.update(name.strip() for name in args.unlabeled_dataset if name.strip())
    return names


def args_for_dataset(args, is_unlabeled):
    if not is_unlabeled:
        return args

    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.label_key = "__ignore_labels_for_unlabeled_domain__"
    dataset_args.keep_not_scored = True
    return dataset_args


def write_embedding_csv(path, embedding, domain_ids, stages, subjects, source_paths, epoch_indices, names):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset",
            "subject",
            "stage",
            "stage_name",
            "epoch_index",
            "source_path",
            "tsne_1",
            "tsne_2",
        ])
        for index in range(embedding.shape[0]):
            stage = int(stages[index])
            writer.writerow([
                names[int(domain_ids[index])],
                subjects[index],
                stage,
                dsv.stage_label(stage),
                int(epoch_indices[index]),
                source_paths[index],
                float(embedding[index, 0]),
                float(embedding[index, 1]),
            ])


def plot_multidomain_tsne(path_png, path_pdf, embedding, domain_ids, stages, names, analysis_label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Skipping plots because matplotlib is unavailable: {}".format(exc))
        return False

    colors = [
        "#2563eb",
        "#dc2626",
        "#059669",
        "#7c3aed",
        "#ea580c",
        "#0891b2",
        "#be123c",
    ]
    markers = ["o", "^", "s", "D", "P", "X", "v"]
    stage_colors = {
        -1: "#6b7280",
        0: "#111827",
        1: "#f59e0b",
        2: "#10b981",
        3: "#6366f1",
        4: "#ec4899",
        dsv.UNKNOWN_STAGE: "#9ca3af",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    for domain_id, name in enumerate(names):
        mask = domain_ids == domain_id
        axes[0].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=9,
            c=colors[domain_id % len(colors)],
            marker=markers[domain_id % len(markers)],
            alpha=0.55,
            linewidths=0,
            label=name,
        )
    axes[0].set_title("t-SNE by dataset")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].legend(frameon=False, markerscale=2.0)

    for stage in sorted(np.unique(stages)):
        for domain_id, name in enumerate(names):
            mask = (stages == stage) & (domain_ids == domain_id)
            if not np.any(mask):
                continue
            axes[1].scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=8,
                c=stage_colors.get(int(stage), "#64748b"),
                marker=markers[domain_id % len(markers)],
                alpha=0.42,
                linewidths=0,
                label="{} ({})".format(dsv.stage_label(stage), name),
            )
    axes[1].set_title("t-SNE by sleep stage")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(frameon=False, fontsize=7, markerscale=1.8, ncol=2)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#e5e7eb", linewidth=0.7)

    fig.suptitle("{} multi-domain shift".format(analysis_label), fontsize=13)
    fig.savefig(path_png, dpi=300)
    fig.savefig(path_pdf)
    plt.close(fig)
    return True


def compute_pairwise_mmd(standardized, domain_ids, names, args, rng):
    results = {}
    for i, name_i in enumerate(names):
        for j in range(i + 1, len(names)):
            name_j = names[j]
            x = standardized[domain_ids == i]
            y = standardized[domain_ids == j]
            key = "{}__vs__{}".format(name_i, name_j)
            results[key] = dsv.compute_mmd(x, y, args, rng)
    return results


def write_summary(path, datasets, names, args, tsne_info, pca_info, pairwise_mmd):
    lines = []
    lines.append("Multi-domain {} domain shift summary".format(args.analysis_label))
    lines.append("=" * len(lines[-1]))
    lines.append("")
    lines.append("Analysis label: {}".format(args.analysis_label))
    lines.append("Per-epoch z-score: {}".format(bool(args.per_epoch_zscore)))
    lines.append("Feature mode: {}".format(args.feature_mode))
    lines.append("Sample mode: {}".format(args.sample_mode))
    lines.append("Standardized features: {}".format(not bool(args.no_standardize)))
    lines.append("PCA used: {}".format(bool(pca_info.get("used"))))
    lines.append("t-SNE perplexity: {:.6f}".format(tsne_info["perplexity"]))
    lines.append("")
    for domain in datasets:
        lines.append("{} available epochs: {}".format(domain["name"], domain["n_available_epochs"]))
        lines.append("{} sampled epochs: {}".format(domain["name"], domain["n_sampled_epochs"]))
        lines.append("{} label policy: {}".format(domain["name"], domain.get("label_policy", "from_npz")))
        lines.append("{} sampled stage counts: {}".format(
            domain["name"], dsv.stringify_counts(domain["stage_counts_sampled"])
        ))
    lines.append("")
    lines.append("Pairwise overall MMD2:")
    for key, result in sorted(pairwise_mmd.items()):
        p_value = result.get("p_value")
        if p_value is None:
            lines.append("  {}: {}".format(key, result.get("mmd2", "NA")))
        else:
            lines.append("  {}: {}, p={:.6f}".format(key, result.get("mmd2", "NA"), p_value))

    labeled_names = []
    for domain in datasets:
        sampled_keys = set(int(key) for key in domain["stage_counts_sampled"].keys())
        if sampled_keys and sampled_keys not in ({dsv.UNKNOWN_STAGE}, {-1}):
            labeled_names.append(domain["name"])
    if len(labeled_names) < len(names):
        lines.append("")
        lines.append("Note: datasets with Unknown or only Not-scored labels are included in overall MMD,")
        lines.append("but true stage-wise MMD requires sleep-stage labels or high-confidence pseudo-labels.")

    with open(path, "w") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main():
    args = parse_args()
    specs = args.dataset if args.dataset is not None else DEFAULT_DATASETS
    dataset_specs = [parse_dataset_spec(spec) for spec in specs]
    if len(dataset_specs) < 2:
        raise RuntimeError("At least two --dataset NAME=PATH entries are required.")

    unlabeled_names = unlabeled_dataset_names(args)
    rng = np.random.RandomState(args.random_seed)
    stage_filter = dsv.parse_stage_filter(args.stage_filter)
    os.makedirs(args.out_dir, exist_ok=True)

    domains = []
    loaded = []
    feature_names = None

    for spec in dataset_specs:
        is_unlabeled = spec["name"] in unlabeled_names
        dataset_args = args_for_dataset(args, is_unlabeled)
        dataset_stage_filter = None if is_unlabeled else stage_filter
        label_policy = "forced_unknown" if is_unlabeled else "from_npz"

        if is_unlabeled:
            print("Scanning {} as unlabeled; NPZ labels will be ignored.".format(spec["name"]))
        else:
            print("Scanning {}...".format(spec["name"]))

        domain = dsv.scan_domain(spec["root"], spec["name"], dataset_args, dataset_stage_filter)
        domain["unlabeled"] = bool(is_unlabeled)
        domain["label_policy"] = label_policy

        selected = dsv.select_epoch_indices(domain, dataset_args, rng)
        print("Extracting {} features...".format(spec["name"]))
        data = dsv.load_selected_features(domain, selected, dataset_args)
        if feature_names is None:
            feature_names = data["feature_names"]
        elif feature_names != data["feature_names"]:
            raise RuntimeError("Feature names do not match for {}".format(spec["name"]))
        domains.append(domain)
        loaded.append(data)

    features = np.vstack([item["features"] for item in loaded])
    stages = np.concatenate([item["labels"] for item in loaded])
    subjects = []
    source_paths = []
    epoch_indices = []
    domain_ids = []

    offset = 0
    for domain_id, item in enumerate(loaded):
        n = item["features"].shape[0]
        domain_ids.append(np.full(n, domain_id, dtype=np.int64))
        subjects.extend(item["subjects"])
        source_paths.extend(item["paths"])
        epoch_indices.extend(item["epoch_indices"])
        offset += n
    domain_ids = np.concatenate(domain_ids)
    names = [domain["name"] for domain in domains]

    print("Standardizing/reducing features...")
    standardized, features_for_tsne, scaler_info, pca_info = dsv.standardize_and_reduce(features, args)

    print("Running t-SNE on {} samples...".format(features_for_tsne.shape[0]))
    embedding, tsne_info = dsv.run_tsne(features_for_tsne, args)

    embedding_csv = os.path.join(args.out_dir, "tsne_embedding.csv")
    write_embedding_csv(embedding_csv, embedding, domain_ids, stages, subjects, source_paths, epoch_indices, names)

    plot_written = False
    if not args.no_plots:
        plot_written = plot_multidomain_tsne(
            os.path.join(args.out_dir, "domain_shift_tsne.png"),
            os.path.join(args.out_dir, "domain_shift_tsne.pdf"),
            embedding,
            domain_ids,
            stages,
            names,
            args.analysis_label,
        )

    print("Computing pairwise MMD...")
    pairwise_mmd = compute_pairwise_mmd(standardized, domain_ids, names, args, rng)

    serializable_domains = []
    for domain in domains:
        serializable_domains.append({
            "name": domain["name"],
            "root": domain["root"],
            "n_files": domain["n_files"],
            "n_records": domain["n_records"],
            "n_skipped_empty_files": domain["n_skipped_empty_files"],
            "n_available_epochs": domain["n_available_epochs"],
            "n_sampled_epochs": domain["n_sampled_epochs"],
            "stage_counts_available": dsv.stringify_counts(domain["stage_counts_available"]),
            "stage_counts_sampled": dsv.stringify_counts(domain["stage_counts_sampled"]),
            "unlabeled": bool(domain.get("unlabeled", False)),
            "label_policy": domain.get("label_policy", "from_npz"),
        })

    result = {
        "datasets": serializable_domains,
        "feature_names": feature_names,
        "feature_mode": args.feature_mode,
        "per_epoch_zscore": bool(args.per_epoch_zscore),
        "standardization": scaler_info,
        "pca": pca_info,
        "tsne": tsne_info,
        "pairwise_mmd": pairwise_mmd,
        "outputs": {
            "embedding_csv": embedding_csv,
            "plot_png": os.path.join(args.out_dir, "domain_shift_tsne.png") if plot_written else None,
            "plot_pdf": os.path.join(args.out_dir, "domain_shift_tsne.pdf") if plot_written else None,
        },
    }

    json_path = os.path.join(args.out_dir, "pairwise_mmd_results.json")
    with open(json_path, "w") as handle:
        json.dump(result, handle, indent=2)

    summary_path = os.path.join(args.out_dir, "domain_shift_summary.txt")
    write_summary(summary_path, domains, names, args, tsne_info, pca_info, pairwise_mmd)

    print("Done.")
    print("Embedding CSV: {}".format(embedding_csv))
    print("Pairwise MMD JSON: {}".format(json_path))
    print("Summary: {}".format(summary_path))
    if plot_written:
        print("t-SNE plot: {}".format(os.path.join(args.out_dir, "domain_shift_tsne.png")))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
