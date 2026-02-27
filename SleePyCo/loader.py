import os
import glob
import json
import torch
import numpy as np
from transform import *
from torch.utils.data import Dataset


class EEGDataLoader(Dataset):

    def __init__(self, config, fold, set="train"):

        self.set = set
        self.fold = fold

        self.sr = 100  # 采样率
        self.dset_cfg = config["dataset"]

        self.root_dir = self.dset_cfg["root_dir"]  # 根目录
        self.dset_name = self.dset_cfg["name"]  # 数据集名称
        self.num_splits = self.dset_cfg["num_splits"]  # 划分fold
        self.eeg_channel = self.dset_cfg["eeg_channel"]  # eeg选用通道

        self.seq_len = self.dset_cfg["seq_len"]  # 序列长度
        self.target_idx = self.dset_cfg["target_idx"]  # 目标通道

        self.training_mode = config["training_params"]["mode"]  # 训练模式

        # 数据集路径更改
        self.dataset_path = os.path.join(
            self.root_dir, self.dset_name, "npz"
        )  # 数据集路径
        self.inputs, self.labels, self.epochs = self.split_dataset()  # 划分数据集

        if self.training_mode == 'pretrain' or self.training_mode == 'mix_FreRA':
            self.transform = Compose(
                transforms=[
                    RandomAmplitudeScale(),
                    RandomTimeShift(),
                    RandomDCShift(),
                    RandomZeroMasking(),
                    RandomAdditiveGaussianNoise(),
                    RandomBandStopFilter(),
                ]
            )
            self.two_transform = TwoTransform(self.transform)

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        n_sample = 30 * self.sr * self.seq_len
        file_idx, idx, seq_len = self.epochs[idx]
        inputs = self.inputs[file_idx][idx : idx + seq_len] # (1,3000)

        if self.set == "train":
            if self.training_mode == "pretrain" :  # 预训练模式
                assert seq_len == 1 
                input_a, input_b = self.two_transform(inputs)  # 增强视图
                input_a = torch.from_numpy(input_a).float()
                input_b = torch.from_numpy(input_b).float()
                inputs = [input_a, input_b]
            elif self.training_mode == "mix_FreRA":
                assert seq_len == 1 
                _, input_b = self.two_transform(inputs)  # 增强视图
                input_a = torch.from_numpy(inputs).float()
                input_b = torch.from_numpy(input_b).float()
                inputs = [input_a, input_b]

            elif self.training_mode in ["scratch", "fullyfinetune", "freezefinetune","FreRA"]:
                inputs = inputs.reshape(1, n_sample)
                inputs = torch.from_numpy(inputs).float()
            else:
                raise NotImplementedError
        else:
            if not self.training_mode == "pretrain":
                inputs = inputs.reshape(1, n_sample)
            inputs = torch.from_numpy(inputs).float()

        labels = self.labels[file_idx][idx : idx + seq_len]#对应输入seq_len的数据的标签
        labels = torch.from_numpy(labels).long()
        # 设置target_idx=-1,也就是输入长度为seq_lend的数据时，
        # 取最后一个时间步的标签作为该序列的标签，也就是通过上文信息进行预测
        # 会造成前seq_len-1个时间步无法预测
        labels = labels[self.target_idx] 

        return inputs, labels

    def split_dataset(self):  # 划分数据集

        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        data_fname_list = [
            os.path.basename(x)
            for x in sorted(glob.glob(os.path.join(data_root, "*.npz")))
        ]
        data_fname_dict = {"train": [], "test": [], "val": []}
        # 获取当前脚本所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        if self.dset_name == "AnphySleep" or self.dset_name == "Anphy_sleep":
            json_path = os.path.join(current_dir, "split_idx", "idx_AnphySleep.json")
            with open(json_path, "r") as f:
                split_table = json.load(f)
            split_idx_list = split_table  # 为保持代码一致性，设置split_idx_list变量
        else:
            split_idx_path = os.path.join(
                current_dir, "split_idx", "idx_{}.npy".format(self.dset_name)
            )
            split_idx_list = np.load(split_idx_path, allow_pickle=True)

        # 构建split_idx的绝对路径
        assert len(split_idx_list) == self.num_splits

        if self.dset_name == "Sleep-EDF-2013":
            for i in range(len(data_fname_list)):
                subject_idx = int(
                    data_fname_list[i][3:5]
                )  # 字符串切片，取文件名的第4个到第5个字符（Python切片是左闭右开区间）
                if subject_idx == self.fold - 1:
                    data_fname_dict["test"].append(data_fname_list[i])
                elif subject_idx in split_idx_list[self.fold - 1]:
                    data_fname_dict["val"].append(data_fname_list[i])
                else:
                    data_fname_dict["train"].append(data_fname_list[i])

        elif self.dset_name == "Sleep-EDF-2018":
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])

        elif self.dset_name == "AnphySleep" or self.dset_name == "Anphy_sleep":
            # AnphySleep 数据集的加载处理
            assert (
                len(split_table) == self.num_splits and 1 <= self.fold <= 10
            ), f"fold必须在1~10之间，当前为{self.fold}"
            data_fname_dict[self.set] = split_table[self.fold - 1]["subjects"][self.set]

        elif (
            self.dset_name == "MASS"
            or self.dset_name == "Physio2018"
            or self.dset_name == "SHHS"
        ):
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
                    
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))

        for data_fname in data_fname_dict[self.set]:
            npz_file = np.load(os.path.join(data_root, data_fname))
            inputs.append(npz_file["x"])
            labels.append(npz_file["y"])
            seq_len = self.seq_len
            if self.dset_name == "MASS" and (
                "-02-" in data_fname or "-04-" in data_fname or "-05-" in data_fname
            ):
                seq_len = int(self.seq_len * 1.5)
            for i in range(len(npz_file["y"]) - seq_len + 1):
                epochs.append([file_idx, i, seq_len])
            file_idx += 1

        return inputs, labels, epochs
