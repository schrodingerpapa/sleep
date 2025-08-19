import mne
import numpy as np
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset
import random


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """带通滤波（AASM推荐EEG频段0.3-35Hz）"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


class UnsupervisedEEGDataset(Dataset):
    def __init__(self, eeg_file_path, fs=100, window_sec=30, stride_sec=5):
        """
        无监督学习专用EEG数据集
        :param eeg_file_path: EDF文件路径
        :param fs: 目标采样率（Hz）
        :param window_sec: 窗口长度（秒）
        :param stride_sec: 滑动步长（秒）
        """
        # 1. 加载并预处理原始EEG
        eeg_file_path = eeg_file_path.replace("\\", "/")  # 修正Windows路径错误
        raw = mne.io.read_raw_edf(eeg_file_path, preload=True)
        self.original_fs = int(raw.info['sfreq'])

        # 自动选择第一个EEG通道
        eeg_ch_names = [ch for ch in raw.ch_names if 'EEG' in ch]
        if not eeg_ch_names:
            raise ValueError("No EEG channels found in the EDF file")
        eeg_data = raw.get_data(picks=eeg_ch_names[0])[0]

        # 降采样（如果必要）
        if self.original_fs != fs:
            eeg_data = mne.filter.resample(eeg_data, down=self.original_fs / fs)
            self.fs = fs
        else:
            self.fs = self.original_fs

        # 带通滤波
        self.eeg_data = butter_bandpass_filter(eeg_data, 0.3, 35, self.fs)

        # 2. 动态分割窗口
        self.window_size = int(window_sec * self.fs)
        self.stride = int(stride_sec * self.fs)
        self.segments = []

        for start in range(0, len(self.eeg_data) - self.window_size, self.stride):
            segment = self.eeg_data[start:start + self.window_size]
            self.segments.append(segment)

        # 3. 数据统计
        self.mean = np.mean(self.eeg_data)
        self.std = np.std(self.eeg_data)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        return self._augment_segment(segment)

    def _augment_segment(self, segment):
        """自监督学习专用数据增强"""
        # 标准化
        segment = (segment - self.mean) / (self.std + 1e-8)

        # 随机选择两种增强组合
        aug_types = [
            self._add_noise,
            self._time_warp,
            self._bandstop_filter,
            self._amplitude_scale
        ]
        aug_fn1, aug_fn2 = random.sample(aug_types, 2)

        aug1 = aug_fn1(segment.copy())
        aug2 = aug_fn2(segment.copy())

        # 确保数组是连续的
        aug1 = np.ascontiguousarray(aug1)
        aug2 = np.ascontiguousarray(aug2)

        return torch.FloatTensor(aug1), torch.FloatTensor(aug2)

    def _add_noise(self, x, noise_level=0.1):
        """添加高斯噪声"""
        x += np.random.normal(0, noise_level, len(x))
        return x

    def _time_warp(self, x, max_stretch=0.2):
        """时间扭曲增强"""
        scale = 1.0 + (random.random() * 2 - 1) * max_stretch
        new_len = int(len(x) * scale)

        # 使用NumPy的interp函数处理时间扭曲
        x = np.interp(
            np.linspace(0, len(x) - 1, new_len),
            np.arange(len(x)),
            x
        )

        if scale > 1:  # 截断
            return x[:len(x)]
        else:  # 填充
            return np.pad(x, (0, len(x) - new_len), 'constant')

    def _bandstop_filter(self, x, notch_freqs=[50]):
        """修正后的带阻滤波器"""
        fs = self.fs
        for freq in notch_freqs:
            if random.random() > 0.5:
                # 确保截止频率在Nyquist频率范围内
                low = max(0, freq - 2)
                high = min(fs / 2 - 0.1, freq + 2)
                if low >= high:
                    continue  # 跳过无效频率范围
                b, a = butter(2, [low, high], fs=fs, btype='bandstop')
                x = filtfilt(b, a, x)
        return x.copy()  # 返回连续的数组

    def _amplitude_scale(self, x, scale_range=[0.8, 1.2]):
        """幅度缩放"""
        return x * random.uniform(*scale_range)


# 使用示例
# 使用示例
if __name__ == "__main__":
    # 更改 matplotlib 后端
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    dataset = UnsupervisedEEGDataset("C:/公开数据集/睡眠脑电公开数据集/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 检查数据增强效果
    aug1, aug2 = dataset[0]
    plt.figure(figsize=(12, 4))
    plt.plot(aug1.numpy(), label='Augmentation 1')
    plt.plot(aug2.numpy(), label='Augmentation 2')
    plt.legend()
    plt.title("EEG Segment with Contrastive Augmentations")
    plt.show()