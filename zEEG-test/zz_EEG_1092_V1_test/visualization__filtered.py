import numpy as np
import pandas as pd
import matplotlib
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import pywt  # 添加小波库

matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy import signal
import plotly.express as px

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 参数设置 (已针对500Hz采样率调整)
SAMPLING_RATE = 500  # 采样率500Hz
FILTER_LOW = 1  # 带通滤波低频截止(Hz)
FILTER_HIGH = 40  # 带通滤波高频截止(Hz)
NOTCH_FREQ = 50  # 陷波滤波频率(Hz)

# 1. 数据加载函数
def load_eeg_data(file_path):
    """加载单通道EEG数据"""
    try:
        data = pd.read_csv(file_path, header=None, names=['EEG'])
        print(f"成功加载数据，共 {len(data)} 个样本 ({len(data) / SAMPLING_RATE:.2f}秒)")
        return data['EEG'].values
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

# 2. 数据预处理函数（完整版）
def preprocess_eeg(eeg_data, fs=SAMPLING_RATE):
    """预处理EEG数据：带通滤波+陷波滤波+毛刺去除+小波去噪"""
    # 检查数据有效性
    if not np.all(np.isfinite(eeg_data)):
        print("警告：输入数据中包含无效值（NaN或Inf），已替换为0")
        eeg_data = np.nan_to_num(eeg_data)

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

    # 毛刺去除 - 基于中值滤波
    filtered_data = signal.medfilt(filtered_data, kernel_size=11)

    # 小波去噪
    def wavelet_denoise(data, wavelet='db4', level=1):
        """使用小波变换进行去噪"""
        coeffs = pywt.wavedec(data, wavelet, mode="per")
        threshold = np.std(coeffs[-level]) * np.sqrt(2 * np.log(len(data)))
        coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        return pywt.waverec(coeffs_thresh, wavelet, mode='per')

    # 应用小波去噪
    filtered_data = wavelet_denoise(filtered_data)

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
    frequencies, psd = signal.welch(eeg_data, fs=fs, nperseg=2048)

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, psd)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.xlim(0, 60)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_band_power(eeg_data, fs=SAMPLING_RATE):
    """绘制各频段能量分布"""
    def bandpower(data, sf, band):
        band = np.asarray(band)
        freqs, psd = signal.welch(data, sf, nperseg=2048)
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.trapz(psd[idx], freqs[idx])

    bands = {
        'Delta (0.5-4Hz)': [0.5, 4],
        'Theta (4-8Hz)': [4, 8],
        'Alpha (8-13Hz)': [8, 13],
        'Beta (13-30Hz)': [13, 30],
        'Gamma (30-45Hz)': [30, 45]
    }

    band_powers = {band: bandpower(eeg_data, fs, freq_range) for band, freq_range in bands.items()}

    plt.figure(figsize=(10, 6))
    plt.bar(band_powers.keys(), band_powers.values())
    plt.title(f'EEG Band Power (Fs={fs}Hz)')
    plt.ylabel('Power (μV²)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return band_powers

# 4. 主程序
def main():
    print(f"\n=== 单通道脑电数据分析 (采样率: {SAMPLING_RATE}Hz) ===")

    file_path = r"C:\研究生\睡眠硬件--采集测试\测试数据\test2\single_EEG\fpc_test_静息.csv"
    raw_eeg = load_eeg_data(file_path)

    if raw_eeg is None:
        return

    try:
        start_time = float(input("请输入开始时间(秒): "))
        end_time = float(input("请输入结束时间(秒): "))

        if start_time < 0 or end_time > len(raw_eeg) / SAMPLING_RATE or start_time >= end_time:
            print("无效的时间范围！使用默认范围...")
            start_time = None
            end_time = None
    except ValueError:
        print("输入无效！使用默认范围...")
        start_time = None
        end_time = None

    print("\n正在进行数据预处理...")
    processed_eeg = preprocess_eeg(raw_eeg)

    print("\n正在绘制原始信号...")
    if start_time is not None and end_time is not None:
        plot_time_series(raw_eeg, title=f'Raw EEG Signal (Fs={SAMPLING_RATE}Hz, Time: {start_time}-{end_time}s)',
                         start_time=start_time, end_time=end_time)
    else:
        plot_time_series(raw_eeg, title=f'Raw EEG Signal (Fs={SAMPLING_RATE}Hz)')

    print("正在绘制处理后的信号...")
    if start_time is not None and end_time is not None:
        plot_time_series(processed_eeg,
                         title=f'Processed EEG Signal (Fs={SAMPLING_RATE}Hz, Time: {start_time}-{end_time}s)',
                         start_time=start_time, end_time=end_time)
    else:
        plot_time_series(processed_eeg, title=f'Processed EEG Signal (Fs={SAMPLING_RATE}Hz)')

    print("正在进行频谱分析...")
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * SAMPLING_RATE)
        end_idx = int(end_time * SAMPLING_RATE)
        plot_spectrum(raw_eeg[start_idx:end_idx],
                      title=f'Power Spectral Density (Fs={SAMPLING_RATE}Hz, Time: {start_time}-{end_time}s)')
    else:
        plot_spectrum(raw_eeg, title=f'Power Spectral Density (Fs={SAMPLING_RATE}Hz)')

    print("正在进行频段能量分析...")
    if start_time is not None and end_time is not None:
        band_powers = plot_band_power(raw_eeg[start_idx:end_idx])
    else:
        band_powers = plot_band_power(raw_eeg)

    print("\n各频段能量分布:")
    for band, power in band_powers.items():
        print(f"{band}: {power:.2f} μV²")

    print("正在进行处理后频谱分析...")
    if start_time is not None and end_time is not None:
        plot_spectrum(processed_eeg[start_idx:end_idx],
                      title=f'Processed Power Spectral Density (Fs={SAMPLING_RATE}Hz, Time: {start_time}-{end_time}s)')
    else:
        plot_spectrum(processed_eeg, title=f'Processed Power Spectral Density (Fs={SAMPLING_RATE}Hz)')

    print("正在进行处理后频段能量分析...")
    if start_time is not None and end_time is not None:
        band_powers = plot_band_power(processed_eeg[start_idx:end_idx])
    else:
        band_powers = plot_band_power(processed_eeg)

    print("\n处理后各频段能量分布:")
    for band, power in band_powers.items():
        print(f"{band}: {power:.2f} μV²")

    output_path = 'processed_eeg_data.csv'
    pd.DataFrame(processed_eeg, columns=['Processed_EEG']).to_csv(output_path, index=False)
    print(f"\n处理后的数据已保存到: {output_path}")

if __name__ == "__main__":
    main()