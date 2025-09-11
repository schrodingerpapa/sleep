# -*- coding:utf-8 -*-
import mne
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import warnings
import os
import glob

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


class TorchDataset(Dataset):
    def __init__(self, paths, sfreq, rfreq, scaler: bool = False):
        super().__init__()
        self.x, self.y = self.get_data(paths, sfreq, rfreq, scaler)
        # 将数据转换为PyTorch张量，x为float32类型，y为long类型
        self.x, self.y = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)

    @staticmethod
    def get_data(paths, sfreq, rfreq, scaler_flag):
        """
        从文件路径中加载数据并进行预处理
        Args:
            paths: 数据文件路径列表
            sfreq: 原始采样频率
            rfreq: 重采样频率
            scaler_flag: 是否进行数据缩放
        Returns:
            total_x: 处理后的特征数据
            total_y: 标签数据
        """
        # 创建MNE信息对象，定义EEG通道信息
        info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=['Fp1'])
        # 创建数据缩放器，使用中位数缩放方法
        scaler = mne.decoding.Scaler(info=info, scalings='median')
        total_x, total_y = [], []
        for path in paths:
            data = np.load(path)
            x, y = data['x'], data['y']
            x = np.expand_dims(x, axis=1)
            if scaler_flag:
                x = scaler.fit_transform(x)
            x = mne.EpochsArray(x, info=info)
            x = x.resample(rfreq)
            x = x.get_data().squeeze()
            total_x.append(x)
            total_y.append(y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)
        return total_x, total_y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item])
        y = torch.tensor(self.y[item])
        return x, y
            # 获取数据并去除维度为1的轴


class EEGDataset(Dataset):
        # 合并所有数据
    def __init__(self, config, fold, set='train'):
        super().__init__()
        # 存储配置参数
        self.config = config
        """返回数据集大小"""
        self.fold = fold
        self.set = set

        """获取指定索引的数据样本"""
        # 从配置中获取参数
        self.dset_cfg = config['dataset']
        self.root_dir = self.dset_cfg['root_dir']  # 待修改
        self.dset_name = self.dset_cfg['name']
        self.eeg_channel = self.dset_cfg['eeg_channel']
        self.seq_len = self.dset_cfg['seq_len']
        self.target_idx = self.dset_cfg['target_idx']
        self.k_splits = self.dset_cfg['k_splits']

        # 加载数据
        self.x, self.y = self.get_data()
        # 转换为PyTorch张量
        self.x, self.y = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)

    def get_data(self):
        """加载数据并按照序列长度提取样本"""
        # 获取对应的数据文件列表
        data_files = self.split_dataset()

        total_x, total_y = [], []

        # 处理每个文件
        for file_path in data_files:
            npz_file = np.load(file_path)
            x, y = npz_file['x'], npz_file['y']

            # 处理序列长度（MASS数据集特殊处理）
            seq_len = self.seq_len
            if self.dset_name == 'MASS' and ('-02-' in os.path.basename(file_path) or
                                             '-04-' in os.path.basename(file_path) or
                                             '-05-' in os.path.basename(file_path)):
                seq_len = int(self.seq_len * 1.5)

            # 提取有效样本
            for i in range(len(y) - seq_len + 1):
                x_segment = x[i:i + seq_len]
                y_segment = y[i:i + seq_len][self.target_idx]  # 提取目标标签

                total_x.append(x_segment)
                total_y.append(y_segment)

        # 合并所有数据
        return np.concatenate(total_x), np.concatenate(total_y)

    def split_dataset(self):  # 划分数据集
        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load(os.path.join('./split_idx', 'idx_{}.npy'.format(self.dset_name)),
                                 allow_pickle=True)  # 读取划分数据集的索引

        assert len(split_idx_list) == self.k_splits

        if self.dset_name == 'Sleep-EDF-2013':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx == self.fold - 1:
                    data_fname_dict['test'].append(data_fname_list[i])
                elif subject_idx in split_idx_list[self.fold - 1]:
                    data_fname_dict['val'].append(data_fname_list[i])
                else:
                    data_fname_dict['train'].append(data_fname_list[i])

        elif self.dset_name == 'Sleep-EDF-2018':
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])

        elif self.dset_name == 'MASS' or self.dset_name == 'Physio2018' or self.dset_name == 'SHHS':
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
        else:
            raise NameError("dataset '{}' cannot be found.".format(self.dataset))

        for data_fname in data_fname_dict[self.set]:
            npz_file = np.load(os.path.join(data_root, data_fname))
            inputs.append(npz_file['x'])
            labels.append(npz_file['y'])
            seq_len = self.seq_len
            if self.dset_name == 'MASS' and ('-02-' in data_fname or '-04-' in data_fname or '-05-' in data_fname):
                seq_len = int(self.seq_len * 1.5)
            for i in range(len(npz_file['y']) - seq_len + 1):
                epochs.append([file_idx, i, seq_len])
            file_idx += 1

        return inputs, labels, epochs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]