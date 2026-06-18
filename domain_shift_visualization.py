#!/usr/bin/env python
"""Domain-shift visualization for two FP1-FP2 NPZ datasets.

The script expects per-record NPZ files with:
  x: epochs, usually shaped (n_epochs, 3000)
  y: sleep-stage labels, usually 0..4 and optionally -1
  fs: sampling rate scalar, optional and defaulting to 100 Hz

It writes a t-SNE embedding plot/CSV and MMD statistics for the two domains.
"""

from __future__ import print_function

import argparse
import csv
import glob
import inspect
import json
import math
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional for this script.
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_SOURCE_ROOT = "/document/chenlungan/datasets/DODH/npz/Fp1-Fp2"
DEFAULT_TARGET_ROOT = "/document/chenlungan/datasets/AnphySleep/npz/Fp1-Fp2"
UNKNOWN_STAGE = -999
EPS = 1e-12

STAGE_NAMES = {
    -1: "Not scored",
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    UNKNOWN_STAGE: "Unknown",
}

BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
    ("low_gamma", 30.0, 45.0),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize domain shift between two FP1-FP2 NPZ datasets using t-SNE and MMD."
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT,
                        help="Directory containing source-domain .npz files.")
    parser.add_argument("--target-root", default=DEFAULT_TARGET_ROOT,
                        help="Directory containing target-domain .npz files.")
    parser.add_argument("--source-name", default="DODH",
                        help="Name used for the source domain in outputs.")
    parser.add_argument("--target-name", default="AnphySleep",
                        help="Name used for the target domain in outputs.")
    parser.add_argument("--out-dir", default=os.path.join("outputs", "domain_shift_fp1_fp2"),
                        help="Directory where plots and statistics will be written.")

    parser.add_argument("--signal-key", default="x", help="NPZ key for epoch signals.")
    parser.add_argument("--label-key", default="y", help="NPZ key for epoch labels.")
    parser.add_argument("--fs-key", default="fs", help="NPZ key for sampling rate.")
    parser.add_argument("--default-fs", type=float, default=100.0,
                        help="Sampling rate used when an NPZ file has no fs key.")
    parser.add_argument("--keep-not-scored", action="store_true",
                        help="Keep y=-1 epochs. By default they are removed.")
    parser.add_argument("--stage-filter", default=None,
                        help="Comma-separated labels to keep, for example 0,1,2,3,4.")

    parser.add_argument("--sample-mode", choices=["balanced", "random", "all"],
                        default="balanced",
                        help="Epoch sampling strategy before feature extraction.")
    parser.add_argument("--max-epochs-per-domain", type=int, default=3000,
                        help="Maximum epochs sampled per domain. Use 0 with --sample-mode all for all epochs.")
    parser.add_argument("--random-seed", type=int, default=2026,
                        help="Random seed for sampling, t-SNE, and MMD permutations.")

    parser.add_argument("--feature-mode", choices=["summary", "raw"], default="summary",
                        help="summary uses time/frequency features; raw downsamples waveform epochs.")
    parser.add_argument("--raw-points", type=int, default=300,
                        help="Number of evenly spaced waveform points for --feature-mode raw.")
    parser.add_argument("--per-epoch-zscore", action="store_true",
                        help="Z-score each epoch before feature extraction.")
    parser.add_argument("--chunk-size", type=int, default=2048,
                        help="Epoch chunk size for feature extraction.")
    parser.add_argument("--no-standardize", action="store_true",
                        help="Do not standardize feature columns before t-SNE/MMD.")

    parser.add_argument("--pca-components", type=int, default=30,
                        help="PCA dimensions before t-SNE. Use 0 to skip PCA.")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0,
                        help="t-SNE perplexity; automatically clipped for small samples.")
    parser.add_argument("--tsne-iterations", type=int, default=1000,
                        help="Number of t-SNE optimization iterations.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip PNG/PDF plot generation and only write CSV/JSON outputs.")

    parser.add_argument("--mmd-kernel", choices=["rbf", "linear"], default="rbf",
                        help="Kernel used for MMD.")
    parser.add_argument("--mmd-sigmas", default="median",
                        help="RBF sigmas, comma-separated, or 'median' for median heuristic.")
    parser.add_argument("--mmd-max-per-domain", type=int, default=1000,
                        help="Maximum samples per domain for each MMD computation.")
    parser.add_argument("--mmd-permutations", type=int, default=200,
                        help="Permutation count for MMD p-values. Use 0 to skip p-values.")
    parser.add_argument("--min-stage-mmd", type=int, default=50,
                        help="Minimum samples per domain needed for stage-wise MMD.")
    return parser.parse_args()


def parse_stage_filter(value):
    if value is None or value == "":
        return None
    stages = []
    for item in value.split(","):
        item = item.strip()
        if item:
            stages.append(int(item))
    return set(stages)


def scalar_from_npz(npz_file, key, default):
    if key not in npz_file.files:
        return default
    value = np.asarray(npz_file[key])
    if value.shape == ():
        return float(value)
    return float(value.reshape(-1)[0])


def prepare_signal_array(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)

    squeezed = np.squeeze(x)
    if squeezed.ndim == 1:
        return squeezed.reshape(1, -1)
    if squeezed.ndim == 2:
        return squeezed
    return squeezed.reshape(squeezed.shape[0], -1)


def stage_label(stage):
    stage = int(stage)
    if stage in STAGE_NAMES:
        return STAGE_NAMES[stage]
    return str(stage)


def stringify_counts(counts):
    return {str(int(key)): int(value) for key, value in sorted(counts.items())}


def scan_domain(root, domain_name, args, stage_filter):
    files = sorted(glob.glob(os.path.join(root, "*.npz")))
    if not files:
        raise RuntimeError("No .npz files found under {}".format(root))

    records = []
    total_valid = 0
    stage_counts = {}
    skipped_empty = 0

    for path in tqdm(files, desc="Scanning {}".format(domain_name)):
        with np.load(path, allow_pickle=True) as npz_file:
            if args.signal_key not in npz_file.files:
                raise KeyError("{} has no '{}' key".format(path, args.signal_key))

            if args.label_key in npz_file.files:
                y = np.asarray(npz_file[args.label_key]).reshape(-1).astype(np.int64)
                n_epochs = len(y)
            else:
                x = prepare_signal_array(npz_file[args.signal_key])
                n_epochs = x.shape[0]
                y = np.full(n_epochs, UNKNOWN_STAGE, dtype=np.int64)

            valid = np.arange(n_epochs, dtype=np.int64)
            if not args.keep_not_scored:
                valid = valid[y[valid] != -1]
            if stage_filter is not None:
                mask = np.zeros(len(valid), dtype=bool)
                for stage in stage_filter:
                    mask |= (y[valid] == stage)
                valid = valid[mask]

            if len(valid) == 0:
                skipped_empty += 1
                continue

            counts = {}
            for stage in np.unique(y[valid]):
                count = int(np.sum(y[valid] == stage))
                counts[int(stage)] = count
                stage_counts[int(stage)] = stage_counts.get(int(stage), 0) + count

            records.append({
                "path": path,
                "subject": os.path.splitext(os.path.basename(path))[0],
                "y": y,
                "fs": scalar_from_npz(npz_file, args.fs_key, args.default_fs),
                "valid_indices": valid,
                "stage_counts": counts,
            })
            total_valid += len(valid)

    if not records:
        raise RuntimeError("No usable epochs found under {}".format(root))

    return {
        "name": domain_name,
        "root": root,
        "records": records,
        "n_records": len(records),
        "n_files": len(files),
        "n_skipped_empty_files": skipped_empty,
        "n_available_epochs": int(total_valid),
        "stage_counts_available": stage_counts,
    }


def allocate_balanced_quotas(stage_counts, max_total):
    stages = [stage for stage, count in sorted(stage_counts.items()) if count > 0]
    total = int(sum(stage_counts.values()))
    if max_total <= 0 or total <= max_total:
        return {stage: int(stage_counts[stage]) for stage in stages}

    quotas = {stage: 0 for stage in stages}
    remaining = int(max_total)

    while remaining > 0:
        open_stages = [stage for stage in stages if quotas[stage] < stage_counts[stage]]
        if not open_stages:
            break
        add_each = max(1, remaining // len(open_stages))
        progressed = False
        for stage in open_stages:
            room = stage_counts[stage] - quotas[stage]
            add = min(room, add_each, remaining)
            if add > 0:
                quotas[stage] += int(add)
                remaining -= int(add)
                progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    return quotas


def add_selected(selected, record_index, indices):
    if len(indices) == 0:
        return
    if record_index not in selected:
        selected[record_index] = []
    selected[record_index].extend([int(index) for index in indices])


def select_by_global_positions(records, total_count, quota, rng, stage=None):
    selected = {}
    if quota <= 0 or total_count <= 0:
        return selected

    quota = min(int(quota), int(total_count))
    chosen_positions = np.sort(rng.choice(total_count, size=quota, replace=False))
    cursor = 0
    position_offset = 0

    for record_index, record in enumerate(records):
        if stage is None:
            local_indices = record["valid_indices"]
        else:
            y = record["y"]
            local_indices = record["valid_indices"][y[record["valid_indices"]] == stage]

        n_local = len(local_indices)
        if n_local == 0:
            continue

        right = position_offset + n_local
        while cursor < len(chosen_positions) and chosen_positions[cursor] < position_offset:
            cursor += 1
        start_cursor = cursor
        while cursor < len(chosen_positions) and chosen_positions[cursor] < right:
            cursor += 1

        if cursor > start_cursor:
            local_positions = chosen_positions[start_cursor:cursor] - position_offset
            add_selected(selected, record_index, local_indices[local_positions])
        position_offset = right

    return selected


def merge_selected(target, source):
    for record_index, indices in source.items():
        add_selected(target, record_index, indices)


def select_epoch_indices(domain, args, rng):
    records = domain["records"]
    total = domain["n_available_epochs"]
    max_epochs = int(args.max_epochs_per_domain)

    if args.sample_mode == "all" or max_epochs <= 0 or total <= max_epochs:
        selected = {}
        for record_index, record in enumerate(records):
            add_selected(selected, record_index, record["valid_indices"])
    elif args.sample_mode == "random":
        selected = select_by_global_positions(records, total, max_epochs, rng, stage=None)
    else:
        selected = {}
        quotas = allocate_balanced_quotas(domain["stage_counts_available"], max_epochs)
        for stage, quota in quotas.items():
            stage_total = domain["stage_counts_available"][stage]
            stage_selected = select_by_global_positions(records, stage_total, quota, rng, stage=stage)
            merge_selected(selected, stage_selected)

    cleaned = {}
    for record_index, indices in selected.items():
        cleaned[record_index] = np.asarray(sorted(set(indices)), dtype=np.int64)
    return cleaned


def zscore_epochs(x):
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    std[std < EPS] = 1.0
    return (x - mean) / std


def summary_feature_names():
    names = [
        "mean",
        "std",
        "rms",
        "median",
        "p05",
        "p25",
        "p75",
        "p95",
        "skew",
        "kurtosis",
        "zero_crossing_rate",
        "line_length",
        "hjorth_mobility",
        "hjorth_complexity",
        "log_total_power",
    ]
    for band_name, _, _ in BANDS:
        names.append("log_power_{}".format(band_name))
    for band_name, _, _ in BANDS:
        names.append("rel_power_{}".format(band_name))
    names.extend(["spectral_centroid", "spectral_entropy", "dominant_frequency"])
    return names


def raw_feature_names(raw_points):
    return ["raw_{:04d}".format(index) for index in range(raw_points)]


def extract_summary_features(x, fs, chunk_size):
    x = prepare_signal_array(x)
    features = []
    feature_names = summary_feature_names()

    for start in range(0, x.shape[0], chunk_size):
        chunk = np.asarray(x[start:start + chunk_size], dtype=np.float64)
        chunk = np.nan_to_num(chunk)
        n_samples = chunk.shape[1]

        mean = np.mean(chunk, axis=1)
        std = np.std(chunk, axis=1)
        std_safe = std.copy()
        std_safe[std_safe < EPS] = 1.0
        centered = chunk - mean[:, None]
        z = centered / std_safe[:, None]

        rms = np.sqrt(np.mean(chunk * chunk, axis=1))
        percentiles = np.percentile(chunk, [5, 25, 50, 75, 95], axis=1)
        skew = np.mean(z ** 3, axis=1)
        kurtosis = np.mean(z ** 4, axis=1) - 3.0
        zero_crossing = np.mean(np.diff(np.signbit(centered), axis=1) != 0, axis=1)
        diff1 = np.diff(chunk, axis=1)
        diff2 = np.diff(diff1, axis=1)
        line_length = np.mean(np.abs(diff1), axis=1)
        var0 = np.var(chunk, axis=1)
        var1 = np.var(diff1, axis=1)
        var2 = np.var(diff2, axis=1)
        mobility = np.sqrt(var1 / (var0 + EPS))
        complexity = np.sqrt(var2 / (var1 + EPS)) / (mobility + EPS)

        window = np.hanning(n_samples)
        spectrum = np.abs(np.fft.rfft(centered * window[None, :], axis=1)) ** 2
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / float(fs))
        max_freq = min(45.0, float(fs) / 2.0)
        usable = (freqs >= 0.5) & (freqs <= max_freq)
        if np.sum(usable) == 0:
            usable = freqs >= 0.0

        usable_power = spectrum[:, usable]
        usable_freqs = freqs[usable]
        total_power = np.sum(usable_power, axis=1) + EPS
        log_total_power = np.log10(total_power)

        band_powers = []
        for _, low, high in BANDS:
            high = min(high, max_freq)
            band_mask = (freqs >= low) & (freqs < high)
            if np.sum(band_mask) == 0:
                band_power = np.zeros(chunk.shape[0], dtype=np.float64)
            else:
                band_power = np.sum(spectrum[:, band_mask], axis=1)
            band_powers.append(band_power)
        band_powers = np.vstack(band_powers).T
        log_band_powers = np.log10(band_powers + EPS)
        relative_band_powers = band_powers / total_power[:, None]

        spectral_probs = usable_power / total_power[:, None]
        spectral_entropy = -np.sum(spectral_probs * np.log(spectral_probs + EPS), axis=1)
        spectral_entropy = spectral_entropy / math.log(max(2, spectral_probs.shape[1]))
        spectral_centroid = np.sum(usable_power * usable_freqs[None, :], axis=1) / total_power
        dominant_frequency = usable_freqs[np.argmax(usable_power, axis=1)]

        block = np.column_stack([
            mean,
            std,
            rms,
            percentiles[2],
            percentiles[0],
            percentiles[1],
            percentiles[3],
            percentiles[4],
            skew,
            kurtosis,
            zero_crossing,
            line_length,
            mobility,
            complexity,
            log_total_power,
            log_band_powers,
            relative_band_powers,
            spectral_centroid,
            spectral_entropy,
            dominant_frequency,
        ])
        features.append(block)

    return np.vstack(features), feature_names


def extract_raw_features(x, raw_points):
    x = prepare_signal_array(x)
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x)
    n_samples = x.shape[1]

    if raw_points <= 0 or raw_points >= n_samples:
        selected = x
    else:
        indices = np.linspace(0, n_samples - 1, raw_points).astype(np.int64)
        selected = x[:, indices]
    return selected, raw_feature_names(selected.shape[1])


def load_selected_features(domain, selected_by_record, args):
    feature_chunks = []
    labels = []
    subjects = []
    source_paths = []
    epoch_indices = []
    feature_names = None

    iterable = sorted(selected_by_record.items(), key=lambda item: item[0])
    for record_index, selected_indices in tqdm(iterable, desc="Loading {}".format(domain["name"])):
        if len(selected_indices) == 0:
            continue
        record = domain["records"][record_index]
        with np.load(record["path"], allow_pickle=True) as npz_file:
            x = prepare_signal_array(npz_file[args.signal_key])

        if np.max(selected_indices) >= x.shape[0]:
            raise RuntimeError(
                "{} has {} signal epochs but selected index {}".format(
                    record["path"], x.shape[0], int(np.max(selected_indices))
                )
            )

        selected_x = x[selected_indices]
        if args.per_epoch_zscore:
            selected_x = zscore_epochs(selected_x)

        if args.feature_mode == "summary":
            features, names = extract_summary_features(selected_x, record["fs"], args.chunk_size)
        else:
            features, names = extract_raw_features(selected_x, args.raw_points)

        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature names changed while loading {}".format(record["path"]))

        y = record["y"][selected_indices]
        feature_chunks.append(features)
        labels.extend([int(value) for value in y])
        subjects.extend([record["subject"]] * len(selected_indices))
        source_paths.extend([record["path"]] * len(selected_indices))
        epoch_indices.extend([int(value) for value in selected_indices])

    if not feature_chunks:
        raise RuntimeError("No selected epochs for {}".format(domain["name"]))

    features = np.vstack(feature_chunks)
    labels = np.asarray(labels, dtype=np.int64)

    sampled_counts = {}
    for stage in np.unique(labels):
        sampled_counts[int(stage)] = int(np.sum(labels == stage))

    domain["n_sampled_epochs"] = int(features.shape[0])
    domain["stage_counts_sampled"] = sampled_counts

    return {
        "features": features,
        "labels": labels,
        "subjects": subjects,
        "paths": source_paths,
        "epoch_indices": epoch_indices,
        "feature_names": feature_names,
    }


def sanitize_features(features):
    features = np.asarray(features, dtype=np.float64)
    features = np.nan_to_num(features)
    return features


def standardize_and_reduce(features, args):
    features = sanitize_features(features)

    if args.no_standardize:
        standardized = features
        scaler_info = {"standardized": False}
    else:
        scaler = StandardScaler()
        standardized = scaler.fit_transform(features)
        standardized = sanitize_features(standardized)
        scaler_info = {
            "standardized": True,
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
        }

    reduced = standardized
    pca_info = {"used": False}
    max_components = min(standardized.shape[0] - 1, standardized.shape[1])
    if args.pca_components > 0 and standardized.shape[1] > args.pca_components and max_components >= 2:
        n_components = min(args.pca_components, max_components)
        pca = PCA(n_components=n_components, random_state=args.random_seed)
        reduced = pca.fit_transform(standardized)
        pca_info = {
            "used": True,
            "n_components": int(n_components),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        }

    return standardized, reduced, scaler_info, pca_info


def run_tsne(features_for_tsne, args):
    n_samples = features_for_tsne.shape[0]
    if n_samples < 4:
        raise RuntimeError("t-SNE needs at least 4 samples, got {}".format(n_samples))

    perplexity = min(float(args.tsne_perplexity), max(1.0, (n_samples - 1) / 3.0))
    iterations = max(int(args.tsne_iterations), 500)
    if iterations != int(args.tsne_iterations):
        print("Using 500 t-SNE iterations because lower values can stop before stable optimization.")

    params = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": 200.0,
        "init": "pca",
        "random_state": args.random_seed,
        "metric": "euclidean",
        "verbose": 1,
    }

    signature = inspect.signature(TSNE.__init__)
    if "max_iter" in signature.parameters:
        params["max_iter"] = iterations
    else:
        params["n_iter"] = iterations

    if "method" in signature.parameters:
        params["method"] = "barnes_hut"
    if "angle" in signature.parameters:
        params["angle"] = 0.5

    tsne = TSNE(**params)
    embedding = tsne.fit_transform(features_for_tsne)
    return embedding, {"perplexity": float(perplexity), "iterations": int(iterations)}


def pairwise_sq_dists(x, y):
    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    distances = x_norm + y_norm - 2.0 * np.dot(x, y.T)
    return np.maximum(distances, 0.0)


def resolve_rbf_sigmas(z, spec, rng):
    if spec != "median":
        sigmas = [float(item.strip()) for item in spec.split(",") if item.strip()]
        if not sigmas:
            raise ValueError("--mmd-sigmas produced an empty sigma list")
        return sigmas

    n = z.shape[0]
    subset_size = min(n, 1000)
    subset_indices = rng.choice(n, size=subset_size, replace=False)
    subset = z[subset_indices]
    d2 = pairwise_sq_dists(subset, subset)
    upper = d2[np.triu_indices(subset_size, k=1)]
    upper = upper[upper > EPS]
    if len(upper) == 0:
        base = 1.0
    else:
        base = math.sqrt(float(np.median(upper)) / 2.0)
    return [base * multiplier for multiplier in [0.5, 1.0, 2.0, 4.0]]


def rbf_kernel(z, sigmas):
    d2 = pairwise_sq_dists(z, z)
    kernel = np.zeros_like(d2)
    for sigma in sigmas:
        sigma = max(float(sigma), EPS)
        kernel += np.exp(-d2 / (2.0 * sigma * sigma))
    kernel /= float(len(sigmas))
    return kernel


def mmd2_from_kernel(kernel, x_indices, y_indices):
    k_xx = kernel[np.ix_(x_indices, x_indices)].mean()
    k_yy = kernel[np.ix_(y_indices, y_indices)].mean()
    k_xy = kernel[np.ix_(x_indices, y_indices)].mean()
    return float(k_xx + k_yy - 2.0 * k_xy)


def linear_mmd2(x, y):
    diff = np.mean(x, axis=0) - np.mean(y, axis=0)
    return float(np.dot(diff, diff))


def sample_for_mmd(x, y, max_per_domain, rng):
    if max_per_domain > 0 and x.shape[0] > max_per_domain:
        x_indices = rng.choice(x.shape[0], size=max_per_domain, replace=False)
        x = x[x_indices]
    if max_per_domain > 0 and y.shape[0] > max_per_domain:
        y_indices = rng.choice(y.shape[0], size=max_per_domain, replace=False)
        y = y[y_indices]
    return x, y


def compute_mmd(x, y, args, rng):
    x, y = sample_for_mmd(x, y, args.mmd_max_per_domain, rng)
    result = {
        "n_source": int(x.shape[0]),
        "n_target": int(y.shape[0]),
        "kernel": args.mmd_kernel,
        "permutations": int(args.mmd_permutations),
    }

    if x.shape[0] < 2 or y.shape[0] < 2:
        result["skipped"] = True
        result["reason"] = "Need at least two samples per domain."
        return result

    if args.mmd_kernel == "linear":
        observed = linear_mmd2(x, y)
        result["mmd2"] = observed
        if args.mmd_permutations > 0:
            z = np.vstack([x, y])
            n_x = x.shape[0]
            n_total = z.shape[0]
            null_stats = []
            for _ in range(args.mmd_permutations):
                perm = rng.permutation(n_total)
                xp = z[perm[:n_x]]
                yp = z[perm[n_x:]]
                null_stats.append(linear_mmd2(xp, yp))
            null_stats = np.asarray(null_stats)
            result["p_value"] = float((np.sum(null_stats >= observed) + 1.0) / (len(null_stats) + 1.0))
            result["null_mean"] = float(np.mean(null_stats))
            result["null_std"] = float(np.std(null_stats))
        return result

    z = np.vstack([x, y])
    sigmas = resolve_rbf_sigmas(z, args.mmd_sigmas, rng)
    kernel = rbf_kernel(z, sigmas)
    n_x = x.shape[0]
    n_total = z.shape[0]
    x_indices = np.arange(n_x)
    y_indices = np.arange(n_x, n_total)
    observed = mmd2_from_kernel(kernel, x_indices, y_indices)

    result["mmd2"] = observed
    result["sigmas"] = [float(value) for value in sigmas]

    if args.mmd_permutations > 0:
        null_stats = []
        for _ in range(args.mmd_permutations):
            perm = rng.permutation(n_total)
            xp = perm[:n_x]
            yp = perm[n_x:]
            null_stats.append(mmd2_from_kernel(kernel, xp, yp))
        null_stats = np.asarray(null_stats)
        result["p_value"] = float((np.sum(null_stats >= observed) + 1.0) / (len(null_stats) + 1.0))
        result["null_mean"] = float(np.mean(null_stats))
        result["null_std"] = float(np.std(null_stats))

    return result


def compute_all_mmd(features, domain_labels, stages, args, rng):
    source_features = features[domain_labels == 0]
    target_features = features[domain_labels == 1]
    results = {
        "overall": compute_mmd(source_features, target_features, args, rng),
        "by_stage": {},
    }

    common_stages = sorted(set(stages[domain_labels == 0]).intersection(set(stages[domain_labels == 1])))
    for stage in common_stages:
        source_stage = features[(domain_labels == 0) & (stages == stage)]
        target_stage = features[(domain_labels == 1) & (stages == stage)]
        key = str(int(stage))
        if min(source_stage.shape[0], target_stage.shape[0]) < args.min_stage_mmd:
            results["by_stage"][key] = {
                "stage_name": stage_label(stage),
                "skipped": True,
                "reason": "Need at least {} samples per domain.".format(args.min_stage_mmd),
                "n_source": int(source_stage.shape[0]),
                "n_target": int(target_stage.shape[0]),
            }
        else:
            value = compute_mmd(source_stage, target_stage, args, rng)
            value["stage_name"] = stage_label(stage)
            results["by_stage"][key] = value
    return results


def write_embedding_csv(path, embedding, domain_labels, stages, subjects, source_paths, epoch_indices,
                        source_name, target_name):
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
            dataset_name = source_name if int(domain_labels[index]) == 0 else target_name
            stage = int(stages[index])
            writer.writerow([
                dataset_name,
                subjects[index],
                stage,
                stage_label(stage),
                int(epoch_indices[index]),
                source_paths[index],
                float(embedding[index, 0]),
                float(embedding[index, 1]),
            ])


def plot_tsne(path_png, path_pdf, embedding, domain_labels, stages, source_name, target_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Skipping plots because matplotlib is unavailable: {}".format(exc))
        return False

    domain_colors = {0: "#2563eb", 1: "#dc2626"}
    stage_colors = {
        -1: "#6b7280",
        0: "#111827",
        1: "#f59e0b",
        2: "#10b981",
        3: "#6366f1",
        4: "#ec4899",
        UNKNOWN_STAGE: "#9ca3af",
    }
    marker_by_domain = {0: "o", 1: "^"}
    name_by_domain = {0: source_name, 1: target_name}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    for domain_id in [0, 1]:
        mask = domain_labels == domain_id
        axes[0].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=9,
            c=domain_colors[domain_id],
            marker=marker_by_domain[domain_id],
            alpha=0.55,
            linewidths=0,
            label=name_by_domain[domain_id],
        )
    axes[0].set_title("t-SNE by dataset")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    axes[0].legend(frameon=False, markerscale=2.0)

    for stage in sorted(np.unique(stages)):
        for domain_id in [0, 1]:
            mask = (stages == stage) & (domain_labels == domain_id)
            if not np.any(mask):
                continue
            label = "{} ({})".format(stage_label(stage), name_by_domain[domain_id])
            axes[1].scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=8,
                c=stage_colors.get(int(stage), "#64748b"),
                marker=marker_by_domain[domain_id],
                alpha=0.45,
                linewidths=0,
                label=label,
            )
    axes[1].set_title("t-SNE by sleep stage")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].legend(frameon=False, fontsize=7, markerscale=1.8, ncol=2)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#e5e7eb", linewidth=0.7)

    fig.suptitle("{} vs {} domain shift on FP1-FP2".format(source_name, target_name), fontsize=13)
    fig.savefig(path_png, dpi=300)
    fig.savefig(path_pdf)
    plt.close(fig)
    return True


def write_summary_text(path, args, source_domain, target_domain, tsne_info, pca_info, mmd_results):
    lines = []
    lines.append("Domain shift visualization summary")
    lines.append("==================================")
    lines.append("")
    lines.append("Source: {} ({})".format(args.source_name, args.source_root))
    lines.append("Target: {} ({})".format(args.target_name, args.target_root))
    lines.append("Feature mode: {}".format(args.feature_mode))
    lines.append("Sample mode: {}".format(args.sample_mode))
    lines.append("Per-epoch z-score: {}".format(bool(args.per_epoch_zscore)))
    lines.append("Standardized features: {}".format(not bool(args.no_standardize)))
    lines.append("PCA used: {}".format(bool(pca_info.get("used"))))
    if pca_info.get("used"):
        lines.append("PCA components: {}".format(pca_info["n_components"]))
        lines.append("PCA variance ratio sum: {:.6f}".format(pca_info["explained_variance_ratio_sum"]))
    lines.append("t-SNE perplexity: {:.6f}".format(tsne_info["perplexity"]))
    lines.append("")
    lines.append("{} available epochs: {}".format(args.source_name, source_domain["n_available_epochs"]))
    lines.append("{} sampled epochs: {}".format(args.source_name, source_domain["n_sampled_epochs"]))
    lines.append("{} sampled stage counts: {}".format(args.source_name, stringify_counts(source_domain["stage_counts_sampled"])))
    lines.append("{} available epochs: {}".format(args.target_name, target_domain["n_available_epochs"]))
    lines.append("{} sampled epochs: {}".format(args.target_name, target_domain["n_sampled_epochs"]))
    lines.append("{} sampled stage counts: {}".format(args.target_name, stringify_counts(target_domain["stage_counts_sampled"])))
    lines.append("")
    overall = mmd_results["overall"]
    lines.append("Overall MMD2: {}".format(overall.get("mmd2", "NA")))
    if "p_value" in overall:
        lines.append("Overall permutation p-value: {:.6f}".format(overall["p_value"]))
    lines.append("")
    lines.append("Stage-wise MMD2:")
    for stage, result in sorted(mmd_results["by_stage"].items(), key=lambda item: int(item[0])):
        if result.get("skipped"):
            lines.append("  {} {}: skipped ({})".format(stage, result.get("stage_name", ""), result.get("reason", "")))
        else:
            p_value = result.get("p_value", None)
            if p_value is None:
                lines.append("  {} {}: {}".format(stage, result.get("stage_name", ""), result.get("mmd2", "NA")))
            else:
                lines.append("  {} {}: {}, p={:.6f}".format(
                    stage, result.get("stage_name", ""), result.get("mmd2", "NA"), p_value
                ))

    with open(path, "w") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main():
    args = parse_args()
    rng = np.random.RandomState(args.random_seed)
    stage_filter = parse_stage_filter(args.stage_filter)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Scanning datasets...")
    source_domain = scan_domain(args.source_root, args.source_name, args, stage_filter)
    target_domain = scan_domain(args.target_root, args.target_name, args, stage_filter)

    print("Selecting epochs...")
    source_selected = select_epoch_indices(source_domain, args, rng)
    target_selected = select_epoch_indices(target_domain, args, rng)

    print("Extracting features...")
    source_loaded = load_selected_features(source_domain, source_selected, args)
    target_loaded = load_selected_features(target_domain, target_selected, args)
    if source_loaded["feature_names"] != target_loaded["feature_names"]:
        raise RuntimeError("Source and target feature names do not match.")

    features = np.vstack([source_loaded["features"], target_loaded["features"]])
    stages = np.concatenate([source_loaded["labels"], target_loaded["labels"]])
    domain_labels = np.concatenate([
        np.zeros(source_loaded["features"].shape[0], dtype=np.int64),
        np.ones(target_loaded["features"].shape[0], dtype=np.int64),
    ])
    subjects = source_loaded["subjects"] + target_loaded["subjects"]
    source_paths = source_loaded["paths"] + target_loaded["paths"]
    epoch_indices = source_loaded["epoch_indices"] + target_loaded["epoch_indices"]

    print("Standardizing/reducing features...")
    standardized, features_for_tsne, scaler_info, pca_info = standardize_and_reduce(features, args)

    print("Running t-SNE on {} samples...".format(features_for_tsne.shape[0]))
    embedding, tsne_info = run_tsne(features_for_tsne, args)

    embedding_csv = os.path.join(args.out_dir, "tsne_embedding.csv")
    write_embedding_csv(
        embedding_csv,
        embedding,
        domain_labels,
        stages,
        subjects,
        source_paths,
        epoch_indices,
        args.source_name,
        args.target_name,
    )

    plot_written = False
    if not args.no_plots:
        plot_written = plot_tsne(
            os.path.join(args.out_dir, "domain_shift_tsne.png"),
            os.path.join(args.out_dir, "domain_shift_tsne.pdf"),
            embedding,
            domain_labels,
            stages,
            args.source_name,
            args.target_name,
        )

    print("Computing MMD...")
    mmd_results = compute_all_mmd(standardized, domain_labels, stages, args, rng)

    result = {
        "source": {
            "name": args.source_name,
            "root": args.source_root,
            "n_files": source_domain["n_files"],
            "n_records": source_domain["n_records"],
            "n_skipped_empty_files": source_domain["n_skipped_empty_files"],
            "n_available_epochs": source_domain["n_available_epochs"],
            "n_sampled_epochs": source_domain["n_sampled_epochs"],
            "stage_counts_available": stringify_counts(source_domain["stage_counts_available"]),
            "stage_counts_sampled": stringify_counts(source_domain["stage_counts_sampled"]),
        },
        "target": {
            "name": args.target_name,
            "root": args.target_root,
            "n_files": target_domain["n_files"],
            "n_records": target_domain["n_records"],
            "n_skipped_empty_files": target_domain["n_skipped_empty_files"],
            "n_available_epochs": target_domain["n_available_epochs"],
            "n_sampled_epochs": target_domain["n_sampled_epochs"],
            "stage_counts_available": stringify_counts(target_domain["stage_counts_available"]),
            "stage_counts_sampled": stringify_counts(target_domain["stage_counts_sampled"]),
        },
        "feature_names": source_loaded["feature_names"],
        "feature_mode": args.feature_mode,
        "per_epoch_zscore": bool(args.per_epoch_zscore),
        "standardization": scaler_info,
        "pca": pca_info,
        "tsne": tsne_info,
        "mmd": mmd_results,
        "outputs": {
            "embedding_csv": embedding_csv,
            "plot_png": os.path.join(args.out_dir, "domain_shift_tsne.png") if plot_written else None,
            "plot_pdf": os.path.join(args.out_dir, "domain_shift_tsne.pdf") if plot_written else None,
        },
    }

    mmd_json = os.path.join(args.out_dir, "mmd_results.json")
    with open(mmd_json, "w") as handle:
        json.dump(result, handle, indent=2)

    summary_txt = os.path.join(args.out_dir, "domain_shift_summary.txt")
    write_summary_text(summary_txt, args, source_domain, target_domain, tsne_info, pca_info, mmd_results)

    print("Done.")
    print("Embedding CSV: {}".format(embedding_csv))
    print("MMD JSON: {}".format(mmd_json))
    print("Summary: {}".format(summary_txt))
    if plot_written:
        print("t-SNE plot: {}".format(os.path.join(args.out_dir, "domain_shift_tsne.png")))
    else:
        print("t-SNE plot was not written.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
