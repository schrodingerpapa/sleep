# -*- coding:utf-8 -*-
import mne
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import warnings
import os
import glob

warnings.filterwarnings("ignore", category=DeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


class TorchDataset(Dataset):
    def __init__(self, paths, sfreq, rfreq, scaler: bool = False):
        super().__init__()
        self.x, self.y = self.get_data(paths, sfreq, rfreq, scaler)
        self.x, self.y = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long)

    @staticmethod
    def get_data(paths, sfreq, rfreq, scaler_flag):
        info = mne.create_info(sfreq=sfreq, ch_types='eeg', ch_names=['Fp1'])
        scaler = mne.decoding.Scaler(info=info, scalings='median')
        total_x, total_y = [], []
        for path in paths:
            data = np.load(path)
            x, y = data['x'], data['y']
            x = np.expand_dims(x, axis=1) # 维度扩展->(sample,1)
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

class EEGDataset(Dataset):
    def __init__(self, config, fold, set='train'):
        super().__init__()
        self.dset_name = config['Dataset']['name']
        self.fold = fold
        self.set = set
        self.k_splits = config['Dataset']['k_splits']
        self.eeg_channel = config['Dataset']['eeg_channel']

        self.seq_len = config['Dataset']['seq_len']  # 从配置中读取
        self.target_idx = config['Dataset']['target_idx']
        self.sr = config['Dataset']['sfreq']  # 从配置中读取采样率
        self.dataset_path = os.path.join('/home/chenlungan/公开数据集', self.dset_name,'npz')
        
        print(f"Initializing dataset: {self.dset_name}, fold: {self.fold}, set: {self.set}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"EEG channel: {self.eeg_channel}")
        
        self.inputs, self.labels, self.epochs = self.split_dataset()
        print(f"Dataset loaded with {len(self.epochs)} samples")
        
        # 如果是空数据集，确保返回正确的维度
        self.n_sample = 30 * self.sr * self.seq_len
        
    def split_dataset(self):
        """
        划分数据集并加载数据
        Returns:
            inputs: 特征数据
            labels: 标签数据
            epochs: 样本索引信息
        """
        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.eeg_channel)
        
        # 检查数据根目录是否存在
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data directory does not exist: {data_root}")
        
        print(f"Looking for data in: {data_root}")
        data_fname_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(data_root, '*.npz')))]
        print(f"Found {len(data_fname_list)} .npz files")
        
        if len(data_fname_list) == 0:
            raise ValueError(f"No .npz files found in {data_root}")
        
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_file = os.path.join('/home/chenlungan/算法模型/NeuroNet/split_idx', f'idx_{self.dset_name}.npy')
        
        # 检查索引文件是否存在
        if not os.path.exists(split_idx_file):
            raise FileNotFoundError(f"Split index file not found: {split_idx_file}")
            
        split_idx_list = np.load(split_idx_file, allow_pickle=True)
        print(f"Loaded split index with {len(split_idx_list)} folds")

        assert len(split_idx_list) == self.k_splits

        # 根据数据集名称划分数据
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
            # 检查fold索引是否有效
            if self.fold - 1 >= len(split_idx_list):
                raise ValueError(f"Fold {self.fold} is out of range for {len(split_idx_list)} folds")
                
            fold_data = split_idx_list[self.fold - 1]
            print(f"Fold data keys: {list(fold_data.keys())}")
            
            # 检查set键是否存在
            if self.set not in fold_data:
                raise ValueError(f"Set '{self.set}' not found in fold {self.fold} data")
                
            for i in range(len(data_fname_list)):
                subject_idx = int(data_fname_list[i][3:5])
                if subject_idx in fold_data[self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])

        elif self.dset_name in ['MASS', 'Physio2018', 'SHHS']:
            for i in range(len(data_fname_list)):
                if i in split_idx_list[self.fold - 1][self.set]:
                    data_fname_dict[self.set].append(data_fname_list[i])
        else:
            raise NameError(f"Dataset '{self.dset_name}' cannot be found.")
            
        print(f"Selected files for {self.set}: {len(data_fname_dict[self.set])}")
        if len(data_fname_dict[self.set]) > 0:
            print(f"File list (first 5): {data_fname_dict[self.set][:5]}")

        # 加载数据并处理序列
        for data_fname in data_fname_dict[self.set]:
            file_path = os.path.join(data_root, data_fname)
            print(f"Loading file: {file_path}")
            npz_file = np.load(file_path)
            x, y = npz_file['x'], npz_file['y']
            print(f"  Data shape: x={x.shape}, y={y.shape}")
            seq_len = self.seq_len

            # 动态调整序列长度（针对 MASS 数据集）
            if self.dset_name == 'MASS' and ('-02-' in data_fname or '-04-' in data_fname or '-05-' in data_fname):
                seq_len = int(self.seq_len * 1.5)

            # 提取有效样本
            available_samples = len(y) - seq_len + 1
            print(f"  Available samples: {available_samples}, seq_len: {seq_len}")
            if available_samples > 0:
                inputs.append(x)
                labels.append(y)
                for i in range(available_samples):
                    epochs.append([file_idx, i, seq_len])
                file_idx += 1
            else:
                print(f"  Warning: Not enough data in {data_fname} for seq_len={seq_len}")

        print(f"Total samples loaded: {len(epochs)}")
        return np.array(inputs, dtype=object), np.array(labels, dtype=object), epochs

    def __len__(self):
        return len(self.epochs)

    def __getitem__(self, idx):
        # 如果数据集为空，返回默认数据
        if len(self.epochs) == 0:
            # 返回正确维度的默认数据 [1, n_sample]
            inputs = torch.zeros(1, self.n_sample, dtype=torch.float32)
            labels = torch.tensor(0, dtype=torch.long)
            return inputs, labels
            
        if idx >= len(self.epochs):
            raise IndexError(f"Index {idx} is out of range for dataset with {len(self.epochs)} samples")
            
        file_idx, idx, seq_len = self.epochs[idx]
        
        # 确保索引有效
        if file_idx < len(self.inputs) and len(self.inputs[file_idx]) > 0:
            inputs = self.inputs[file_idx][idx:idx+seq_len]
        else:
            # 如果数据无效，返回默认值
            inputs = np.zeros((seq_len, 3000))
        
        # 检查输入数据是否为空
        if inputs.size == 0:
            inputs = np.zeros((seq_len, 3000))
        
        # 确保数据形状正确 [1, n_sample]
        inputs = inputs.reshape(1, self.n_sample)
        inputs = torch.from_numpy(inputs).float()
        
        # 处理标签数据
        if file_idx < len(self.labels) and len(self.labels[file_idx]) > 0:
            labels = self.labels[file_idx][idx:idx+seq_len]
        else:
            labels = np.array([0] * seq_len)
            
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]
        
        return inputs, labels