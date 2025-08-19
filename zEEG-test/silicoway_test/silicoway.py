import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy import signal
# import plotly.express as px
from scipy.signal import butter, filtfilt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 参数设置 (已针对250Hz采样率调整)
SAMPLING_RATE = 250  # 采样率250Hz
FILTER_LOW = 0.5  # 带通滤波低频截止(Hz)
FILTER_HIGH = 40  # 带通滤波高频截止(Hz) - 略高于通常的30Hz以便观察
NOTCH_FREQ = 50  # 陷波滤波频率(Hz)
VREF = 2.42  # 参考电压
PGA = 6  # 可编程增益放大器增益
gain = 255


# 1. 数据加载函数
import os

def load_eeg_data(file_path):
    raw_data = pd.read_csv(file_path)
    """加载单通道EEG数据，从最后一列的第二行开始，支持csv和xlsx"""
    try:
        eeg_data = raw_data.iloc[1:, -1]  # 读取最后一列，从第二行开始
        eeg = eeg_data*1e6*VREF/(gain*PGA*(2**23))
        #eeg = eeg_data # 将数据转换为微
        return eeg.values
    except Exception as e:  # 捕获所有异常
        print(f"加载数据失败: {e}")
        return None




# 2. 数据预处理函数
def preprocess_eeg(eeg_data, fs=SAMPLING_RATE):
    """预处理EEG数据：带通滤波+陷波滤波"""

    # 带通滤波
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    # 陷波滤波
    def butter_notch(notch_freq, fs, quality_factor=30):
        nyq = 0.5 * fs
        freq = notch_freq / nyq
        b, a = butter(2, [freq - 0.1 / quality_factor, freq + 0.1 / quality_factor], btype='bandstop')
        return b, a

    # 应用带通滤波
    b, a = butter_bandpass(FILTER_LOW, FILTER_HIGH, fs)
    filtered_data = filtfilt(b, a, eeg_data)

    # 应用陷波滤波
    if NOTCH_FREQ < fs / 2:
        b, a = butter_notch(NOTCH_FREQ, fs)
        filtered_data = filtfilt(b, a, filtered_data)

    return filtered_data


# 3. 可视化函数
def plot_time_series(eeg_data, fs=SAMPLING_RATE, title='EEG Signal', start_time=None, end_time=None):
    """绘制时间序列图，可指定开始和结束时间"""
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        eeg_data = eeg_data[start_idx:end_idx]
        time = np.arange(len(eeg_data)) / fs + start_time
    else:
        time = np.arange(len(eeg_data)) / fs

    plt.figure(figsize=(15, 5))
    plt.plot(time, eeg_data, linewidth=0.5)
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (μV)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectrum(eeg_data, fs=SAMPLING_RATE, title='功率谱密度'):
    """绘制功率谱密度图"""
    frequencies, psd = signal.welch(eeg_data, fs=fs, nperseg=2048)  # 增加nperseg提高频率分辨率

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, psd)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.xlim(0, 60)  # 显示到60Hz以便观察高频成分
    # plt.axvline(NOTCH_FREQ, color='r', linestyle='--', alpha=0.3, label=f'Notch @ {NOTCH_FREQ}Hz')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_band_power(eeg_data, fs=SAMPLING_RATE):
    """绘制各频段能量分布"""

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
        'Gamma (30-45Hz)': [30, 45]  # 添加Gamma频段
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

def show_freq(data, Fs=SAMPLING_RATE):
    L = len(data)
    T = 1.0 / Fs  # 采样周期 (s)

    yf = fft(data)
    yf = 2 * np.abs(yf[:(L // 2)] / L)  # 取FFT结果的绝对值
    yf = (yf / np.max(yf)) ** 2

    xf = np.arange(0, (L // 2)) * (Fs / L)

    return xf, yf
def My_plot_freq(xf, yf):
    plt.figure(figsize=(6, 6))
    plt.plot(xf, yf)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


# 4. 主程序
def main():
    print(f"\n=== 单通道脑电数据分析 (采样率: {SAMPLING_RATE}Hz) ===")

    # 加载数据
    file_path = r"C:\Users\clg\Desktop\50μV--无滤波--PGA6.csv"
    raw_eeg = load_eeg_data(file_path)

    if raw_eeg is None:
        return

    # 获取用户输入的时间范围
    try:
        print("数据时长: {:.2f}秒".format(len(raw_eeg) / SAMPLING_RATE))
        start_time = float(input("请输入开始时间(秒): "))
        end_time = float(input("请输入结束时间(秒): "))
        if start_time < 0 or end_time > len(raw_eeg) / SAMPLING_RATE or start_time >= end_time:
            print("无效的时间范围！使用默认范围...")
            start_time = 0
            end_time = len(raw_eeg) / SAMPLING_RATE
    except Exception:
        print("输入无效！使用默认范围...")
        start_time = 0
        end_time = len(raw_eeg) / SAMPLING_RATE

    # 预处理
    print("\n正在进行数据预处理(带通滤波+陷波滤波)...")
    processed_eeg = preprocess_eeg(raw_eeg)

    # 可视化频谱
    xf, yf = show_freq(processed_eeg, 250)
    My_plot_freq(xf, yf)


    # 可视化原始数据
    print("\n正在绘制原始信号...")
    if start_time is not None and end_time is not None:
        plot_time_series(raw_eeg, title=f'Raw EEG Signal (Fs={SAMPLING_RATE}Hz, Time: {start_time}-{end_time}s)',
                         start_time=start_time, end_time=end_time)
    else:
        plot_time_series(raw_eeg, title=f'Raw EEG Signal (Fs={SAMPLING_RATE}Hz)')

    # 可视化处理后的数据
    print("正在绘制处理后的信号...")
    if start_time is not None and end_time is not None:
        plot_time_series(processed_eeg,
                         title=f'Processed EEG Signal (Fs={SAMPLING_RATE}Hz, Time: {start_time}-{end_time}s)',
                         start_time=start_time, end_time=end_time)
    else:
        plot_time_series(processed_eeg, title=f'Processed EEG Signal (Fs={SAMPLING_RATE}Hz)')

    # 原始频谱分析
    print("正在进行频谱分析...")
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * SAMPLING_RATE)
        end_idx = int(end_time * SAMPLING_RATE)
        plot_spectrum(raw_eeg[start_idx:end_idx],
                      title=f'Power Spectral Density (Fs={SAMPLING_RATE}Hz, Time: {start_time}-{end_time}s)')
    else:
        plot_spectrum(raw_eeg, title=f'Power Spectral Density (Fs={SAMPLING_RATE}Hz)')

    # 频段能量分析
    print("正在进行频段能量分析...")
    if start_time is not None and end_time is not None:
        band_powers = plot_band_power(raw_eeg[start_idx:end_idx])
    else:
        band_powers = plot_band_power(raw_eeg)

    print("\n各频段能量分布:")
    for band, power in band_powers.items():
        print(f"{band}: {power:.2f} μV²")



    # 频谱分析
    print("正在进行频谱分析...")
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * SAMPLING_RATE)
        end_idx = int(end_time * SAMPLING_RATE)
        plot_spectrum(processed_eeg[start_idx:end_idx],
                      title=f'Power Spectral Density (Fs={SAMPLING_RATE}Hz, Time: {start_time}-{end_time}s)')
    else:
        plot_spectrum(processed_eeg, title=f'Power Spectral Density (Fs={SAMPLING_RATE}Hz)')

    # 频段能量分析
    print("正在进行频段能量分析...")
    if start_time is not None and end_time is not None:
        band_powers = plot_band_power(processed_eeg[start_idx:end_idx])
    else:
        band_powers = plot_band_power(processed_eeg)

    print("\n各频段能量分布:")
    for band, power in band_powers.items():
        print(f"{band}: {power:.2f} μV²")


    # 保存处理后的数据
    output_path = 'processed_eeg_data.csv'
    pd.DataFrame(processed_eeg, columns=['Processed_EEG']).to_csv(output_path, index=False)
    print(f"\n处理后的数据已保存到: {output_path}")


if __name__ == "__main__":
    main()
