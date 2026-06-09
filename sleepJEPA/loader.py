import glob
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from transforms import Compose, RandomGaussianNoise, RandomTimeShift, RandomZeroMasking, TwoTransform


class EEGJEPADataLoader(Dataset):
    def __init__(self, config: dict, fold: int, set: str = "train"):
        self.config = config
        self.fold = fold
        self.set = set
        self.dset_cfg = config["dataset"]
        self.training_mode = config["training_params"]["mode"]

        self.root_dir = self.dset_cfg["root_dir"]
        self.dataset_name = self.dset_cfg["name"]
        self.eeg_channel = self.dset_cfg["eeg_channel"]
        self.num_splits = self.dset_cfg.get("num_splits", 10)
        self.seq_len = self.dset_cfg.get("seq_len", 1)
        self.target_idx = self.dset_cfg.get("target_idx", 0)
        self.num_classes = self.dset_cfg.get("num_classes", 5)

        self.dataset_path = os.path.join(self.root_dir, self.dataset_name, "npz", self.eeg_channel)
        self.inputs, self.labels, self.epochs = self.split_dataset()

        self.transform = None
        if self.training_mode == "pretrain":
            self.transform = Compose(
                [
                    RandomGaussianNoise(std=0.01),
                    RandomTimeShift(max_shift=50),
                    RandomZeroMasking(mask_ratio=0.1),
                ]
            )
            self.two_transform = TwoTransform(self.transform)

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx: int):
        file_idx, start, seq_len = self.epochs[idx]
        eeg = self.inputs[file_idx][start : start + seq_len]

        if eeg.ndim == 1:
            eeg = eeg[None, :]
        elif eeg.ndim == 2 and eeg.shape[0] != 1:
            eeg = eeg[:1, :]

        if self.training_mode == "pretrain":
            input_a, input_b = self.two_transform(eeg.astype(np.float32))
            return torch.from_numpy(input_a).float(), torch.from_numpy(input_b).float()

        inputs = torch.from_numpy(eeg.astype(np.float32))
        label = self.labels[file_idx][start + self.target_idx]
        return inputs, torch.tensor(int(label), dtype=torch.long)

    def split_dataset(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[int]]]:
        inputs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        epochs: List[List[int]] = []

        if not os.path.isdir(self.dataset_path):
            raise FileNotFoundError(f"EEG data folder not found: {self.dataset_path}")

        file_list = sorted(glob.glob(os.path.join(self.dataset_path, "*.npz")))
        data_names = [os.path.basename(x) for x in file_list]
        split_idx = self.load_split_idx()

        membership = {"train": [], "val": [], "test": []}
        if split_idx is not None:
            if self.dataset_name == "Sleep-EDF-2018":
                for fname in data_names:
                    subject_idx = int(fname[3:5])
                    if subject_idx in split_idx[self.fold - 1][self.set]:
                        membership[self.set].append(fname)
            elif self.dataset_name == "Sleep-EDF-2013":
                for fname in data_names:
                    subject_idx = int(fname[3:5])
                    if subject_idx == self.fold - 1:
                        membership["test"].append(fname)
                    elif subject_idx in split_idx[self.fold - 1]:
                        membership["val"].append(fname)
                    else:
                        membership["train"].append(fname)
            else:
                for fname in data_names:
                    if fname in split_idx[self.fold - 1].get(self.set, []):
                        membership[self.set].append(fname)
        else:
            for idx, fname in enumerate(data_names):
                if idx % self.num_splits == self.fold - 1:
                    membership["test"].append(fname)
                elif idx % self.num_splits == (self.fold % self.num_splits):
                    membership["val"].append(fname)
                else:
                    membership["train"].append(fname)

        for fname in membership[self.set]:
            npz = np.load(os.path.join(self.dataset_path, fname))
            inputs.append(npz["x"])
            labels.append(npz["y"])
            total_steps = len(npz["y"]) - self.seq_len + 1
            for step in range(total_steps):
                epochs.append([len(inputs) - 1, step, self.seq_len])

        return inputs, labels, epochs

    def load_split_idx(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "split_idx", f"idx_{self.dataset_name}.json")
        npy_path = os.path.join(current_dir, "split_idx", f"idx_{self.dataset_name}.npy")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        if os.path.exists(npy_path):
            return np.load(npy_path, allow_pickle=True)
        return None
