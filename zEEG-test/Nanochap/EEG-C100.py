import re
import numpy as np

from typing import Dict, Tuple, List
from scipy import signal

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fs = 250

def read_eeg_file(file_path: str) -> Tuple[Dict[str, str], np.ndarray]:
    """
    读取脑电数据文件，解析头部元数据和两列数据

    参数:
        file_path: 文件路径

    返回:
        元组(metadata, data)，其中 metadata 是包含元数据的字典，data 是包含脑电数据的 numpy 数组
    """
    metadata = {
      'sampling_rate': 250,
        'channels': ['原始数据', '处理后数据']
    }
    data_start_line = 0

    # 读取文件头，提取元数据
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # 解析元数据行
        for i, line in enumerate(lines):
            line = line.strip()

            # 检测时间行 (格式: Time:2025-05-09 11-26-26)
            if line.startswith('Time:'):
                time_str = line.split(':', 1)[1].strip()
                # 将 - 替换为 : 以符合标准时间格式
                time_str = re.sub(r'(\d{2})-(\d{2})-(\d{2})$', r'\1:\2:\3', time_str)
                metadata['time'] = time_str

            # 检测通道行 (格式: channel1  channel2)
            elif 'channel' in line.lower():
                # 取消注释下面一行，如果你想忽略文件中的通道名称
                # channel_names = ['原始数据', '处理后数据']

                # 保留原逻辑，同时确保有两个通道名称
                channel_names = [ch.strip() for ch in line.split() if 'channel' in ch.lower()]
                if len(channel_names) >= 2:
                    metadata['channels'] = ['原始数据', '处理后数据']
                data_start_line = i + 1
                break

    # 读取两列数据
    try:
        data = np.loadtxt(file_path, skiprows=data_start_line, usecols=(0, 1))
    except Exception as e:
        print(f"读取数据时出错: {e}")
        print("尝试使用逗号作为分隔符...")
        data = np.loadtxt(file_path, skiprows=data_start_line, usecols=(0, 1), delimiter=',')

    return metadata, data


def visualize_eeg(data: np.ndarray, metadata: Dict[str, str], title: str = "EEG Data", start_time=None, end_time=None):
    sampling_rate = metadata.get('sampling_rate', fs)
    channel_names = metadata.get('channels', ['原始数据', '处理后后数据'])

    if start_time is not None and end_time is not None:
        start_idx = int(start_time * sampling_rate)
        end_idx = int(end_time * sampling_rate)
        data = data[start_idx:end_idx]
        time = np.arange(len(data)) / sampling_rate + start_time
    else:
        time = np.arange(data.shape[0]) / sampling_rate

    plt.figure(figsize=(14, 8))
    plt.suptitle(title, fontsize=16)

    for i, channel in enumerate(channel_names[:data.shape[1]]):
        plt.subplot(data.shape[1], 1, i + 1)
        plt.plot(time, data[:, i], linewidth=0.8)
        plt.title(f"{channel}", fontsize=12)
        plt.xlabel('时间 (秒)', fontsize=10)
        plt.ylabel('幅度', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(pad=3.0)

    plt.show()


def filter_eeg(data: np.ndarray, sampling_rate: float, low_cut: float = 1, high_cut: float = 40.0,
               notch: float = 50.0) -> np.ndarray:
    """
    对脑电数据进行滤波处理

    参数:
        data: 原始脑电数据
        sampling_rate: 采样率
        low_cut: 低通滤波截止频率
        high_cut: 高通滤波截止频率
        notch: 陷波滤波频率 (用于去除工频干扰)

    返回:
        滤波后的脑电数据
    """
    # 1. 带通滤波
    nyquist = 0.5 * sampling_rate
    low, high = low_cut / nyquist, high_cut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data, axis=0)

    # 2. 工频陷波滤波 (50Hz 或 60Hz，根据实际情况选择)
    if notch > 0:
        f0 = notch / nyquist
        Q = 30.0
        b_notch, a_notch = signal.iirnotch(f0, Q)
        filtered_data = signal.filtfilt(b_notch, a_notch, filtered_data, axis=0)

    return filtered_data


def compute_spectrum(data: np.ndarray, sampling_rate: float, segment_length: int = None) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    计算脑电数据的功率谱密度

    参数:
        data: 脑电数据
        sampling_rate: 采样率
        segment_length: FFT分段长度

    返回:
        频率数组和功率谱密度数组
    """
    # 如果未指定segment_length，则使用数据长度的1/4
    if segment_length is None:
        segment_length = min(int(len(data) / 4), 2048)

    freqs, psd = signal.welch(data, sampling_rate, nperseg=segment_length, axis=0)
    return freqs, psd


def visualize_spectrum(freqs: np.ndarray, psd: np.ndarray, metadata: Dict[str, str], max_freq: float = 50.0):
    """
    可视化脑电数据的功率谱密度

    参数:
        freqs: 频率数组
        psd: 功率谱密度数组
        metadata: 包含元数据的字典
        max_freq: 显示的最大频率
    """
    channel_names = metadata.get('channels', ['原始数据', '处理后数据'])

    # 找到小于max_freq的频率索引
    idx = np.where(freqs <= max_freq)[0]
    if len(idx) == 0:
        idx = [0, len(freqs) - 1]
    else:
        idx = [0, idx[-1]]

    plt.figure(figsize=(14, 6))

    for i, channel in enumerate(channel_names[:psd.shape[1]]):
        plt.subplot(1, psd.shape[1], i + 1)
        plt.plot(freqs[idx[0]:idx[1]+1], psd[idx[0]:idx[1]+1, i])
        plt.title(f"{channel} 功率谱密度", fontsize=12)
        plt.xlabel('频率 (Hz)', fontsize=10)
        plt.ylabel('功率/频率 (dB/Hz)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_band_power(eeg_data, fs=250):
    """绘制各频段能量分布，支持多通道数据"""

    # 检查数据是否为多通道
    if eeg_data.ndim > 1 and eeg_data.shape[1] > 1:
        n_channels = eeg_data.shape[1]
        channel_names = ['原始数据', '处理后后数据'][:n_channels]

        for ch_idx in range(n_channels):
            channel_data = eeg_data[:, ch_idx]

            def bandpower(data, sf, band):
                band = np.asarray(band)
                freqs, psd = signal.welch(data, sf, nperseg=2048)
                idx = np.logical_and(freqs >= band[0], freqs <= band[1])
                return np.trapz(psd[idx], freqs[idx])

            # 定义脑电频段
            bands = {
                'Delta (0.5-4Hz)': [0.5, 4],
                'Theta (4-8Hz)': [4, 8],
                'Alpha (8-13Hz)': [8, 13],
                'Beta (13-30Hz)': [13, 30],
                'Gamma (30-45Hz)': [30, 45]
            }

            # 计算各频段能量
            band_powers = {band: bandpower(channel_data, fs, freq_range)
                           for band, freq_range in bands.items()}

            # 绘制
            plt.figure(figsize=(10, 6))
            plt.bar(band_powers.keys(), band_powers.values())
            plt.title(f'{channel_names[ch_idx]} 各频段能量分布 (Fs={fs}Hz)')
            plt.ylabel('Power (μV²)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            print(f"\n{channel_names[ch_idx]} 各频段能量分布:")
            for band, power in band_powers.items():
                print(f"{band}: {power:.2f} μV²")
    else:
        # 单通道数据处理逻辑（保持原有代码）
        def bandpower(data, sf, band):
            band = np.asarray(band)
            freqs, psd = signal.welch(data, sf, nperseg=2048)
            idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            return np.trapz(psd[idx], freqs[idx])

        # 定义脑电频段
        bands = {
            'Delta (0.5-4Hz)': [0.5, 4],
            'Theta (4-8Hz)': [4, 8],
            'Alpha (8-13Hz)': [8, 13],
            'Beta (13-30Hz)': [13, 30],
            'Gamma (30-45Hz)': [30, 45]
        }

        # 计算各频段能量
        band_powers = {band: bandpower(eeg_data, fs, freq_range)
                       for band, freq_range in bands.items()}

        # 绘制
        plt.figure(figsize=(10, 6))
        plt.bar(band_powers.keys(), band_powers.values())
        plt.title(f'EEG Band Power (Fs={fs}Hz)')
        plt.ylabel('Power (μV²)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return band_powers


def main():
    """主函数，演示如何使用上述函数"""
    file_path = r"C:\研究生\睡眠硬件\暖芯迦C1001脑电模块\测试数据\2025.06.16 15.57.12---信号发生器.txt"


    try:
        # 读取数据
        metadata, data = read_eeg_file(file_path)

        # 打印元数据
        print("元数据:")
        for key, value in metadata.items():
            print(f"{key}: {value}")

        # 打印数据信息
        print(f"\n数据形状: {data.shape}")
        print(f"通道数: {data.shape[1]}")
        print(f"样本数: {data.shape[0]}")
        print(f"数据时长: {data.shape[0] / metadata['sampling_rate']:.2f} 秒")

        # 获取用户输入的时间范围
        try:
            start_time = float(input("请输入开始时间(秒): "))
            end_time = float(input("请输入结束时间(秒): "))

            if start_time < 0 or end_time > data.shape[0] / metadata['sampling_rate'] or start_time >= end_time:
                print("无效的时间范围！使用默认范围...")
                start_time = None
                end_time = None
        except ValueError:
            print("输入无效！使用默认范围...")
            start_time = None
            end_time = None

        # 可视化原始数据
        print("\n显示原始数据...")
        visualize_eeg(data, metadata, title="原始脑电数据", start_time=start_time, end_time=end_time)

        # 数据滤波
        print("\n正在进行信号滤波...")
        filtered_data = filter_eeg(data, metadata['sampling_rate'])

        # 可视化滤波后的数据
        print("\n显示滤波后的数据...")
        visualize_eeg(filtered_data, metadata, title="处理后脑电数据", start_time=start_time, end_time=end_time)

        # 计算并可视化功率谱
        print("\n计算并显示功率谱密度...")
        if start_time is not None and end_time is not None:
            start_idx = int(start_time * metadata['sampling_rate'])
            end_idx = int(end_time * metadata['sampling_rate'])
            filtered_data_selected = filtered_data[start_idx:end_idx]
            freqs, psd = compute_spectrum(filtered_data_selected, metadata['sampling_rate'])
        else:
            freqs, psd = compute_spectrum(filtered_data, metadata['sampling_rate'])
        visualize_spectrum(freqs, psd, metadata)

        # 频段
        print("\n计算各频段能量分布...")
        if start_time is not None and end_time is not None:
            filtered_data_selected = filtered_data[start_idx:end_idx]
            plot_band_power(filtered_data_selected, fs)
        else:
            plot_band_power(filtered_data, fs)

        # 保存数据为 numpy 格式，便于后续处理
        np.savez('eeg_data_processed.npz',
                 raw_data=data,
                 filtered_data=filtered_data,
                 metadata=metadata)
        print("\n数据已保存为 eeg_data_processed.npz")

    except Exception as e:
        print(f"处理数据时出错: {e}")


if __name__ == "__main__":
    main()
