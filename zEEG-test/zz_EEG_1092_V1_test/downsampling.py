import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt, periodogram, welch, butter, sosfilt
matplotlib.use('TkAgg')  # 或者尝试 'QtAgg', 'TkAgg' 等其他后端

# 读取文件
file_path = r"C:\Users\clg\Desktop\测试--SW3011采集数据\8-14睡眠测试_20250814_003648.csv"  # 替换为你的文件路径
data = pd.read_csv(file_path, header=None)  # 假设没有表头
signal = data.values.flatten()  # 将数据转换为一维数组

# 原始采样率和目标采样率
fs_original = 500  # 原始采样率
fs_target = 100    # 目标采样率

# 降采样因子
downsample_factor = int(fs_original / fs_target)

# 降采样后的信号
downsampled_signal = signal[::downsample_factor]

# 绘制原始信号和降采样后的信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(signal)
plt.title('Original Signal (500 Hz)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(downsampled_signal)
plt.title(f'Downsampled Signal ({fs_target} Hz)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# 对降采样后的信号进行进一步处理，例如应用巴特沃斯滤波器
# 添加 0.5-40Hz 的 4 阶巴特沃斯滤波
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    y = sosfilt(sos, data)
    return y

lowcut = 0.5  # 低频截止频率
highcut = 35  # 高频截止频率
butterworth_filtered_signal = butter_bandpass_filter(downsampled_signal, lowcut, highcut, fs_target, order=4)

# 绘制巴特沃斯滤波后的信号
plt.figure(figsize=(10, 4))
plt.plot(butterworth_filtered_signal)
plt.title('Filtered Signal with Butterworth Filter (0.5-40Hz)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# 使用周期图法和 Welch 法对降采样后的信号进行频谱分析
# 使用周期图法计算频谱
f_periodogram, Pxx_periodogram = periodogram(downsampled_signal, fs=fs_target)
f_periodogram_filtered, Pxx_periodogram_filtered = periodogram(butterworth_filtered_signal, fs=fs_target)

# 使用 Welch 法计算频谱
f_welch, Pxx_welch = welch(downsampled_signal, fs=fs_target, window='hann', nperseg=512, noverlap=256)
f_welch_filtered, Pxx_welch_filtered = welch(butterworth_filtered_signal, fs=fs_target, window='hann', nperseg=512, noverlap=256)

# 绘制降采样后的信号的频谱
plt.figure(figsize=(12, 8))

# 降采样后信号频谱（FFT）
plt.subplot(2, 2, 1)
n = len(downsampled_signal)
freq = np.fft.fftfreq(n, 1/fs_target)
fft_spectrum_downsampled = np.abs(np.fft.fft(downsampled_signal)) / n
plt.plot(freq[:n//2], fft_spectrum_downsampled[:n//2])
plt.xlim(0, 50)
plt.title('Downsampled Signal Spectrum (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# 降采样后信号频谱（周期图法）
plt.subplot(2, 2, 2)
plt.semilogy(f_periodogram, Pxx_periodogram)
plt.xlim(0, 50)
plt.title('Downsampled Signal Spectrum (Periodogram)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

# 巴特沃斯滤波后信号频谱（FFT）
plt.subplot(2, 2, 3)
fft_spectrum_butterworth = np.abs(np.fft.fft(butterworth_filtered_signal)) / n
plt.plot(freq[:n//2], fft_spectrum_butterworth[:n//2])
plt.xlim(0, 50)
plt.title('Butterworth Filtered Signal Spectrum (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# 巴特沃斯滤波后信号频谱（周期图法）
plt.subplot(2, 2, 4)
plt.semilogy(f_periodogram_filtered, Pxx_periodogram_filtered)
plt.xlim(0, 50)
plt.title('Butterworth Filtered Signal Spectrum (Periodogram)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.show()

# 绘制降采样后信号和巴特沃斯滤波后信号的 Welch 频谱
plt.figure(figsize=(12, 8))

# 降采样后信号频谱（Welch 法）
plt.subplot(1, 2, 1)
plt.semilogy(f_welch, Pxx_welch)
plt.xlim(0, 50)
plt.title('Downsampled Signal Spectrum (Welch)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

# 巴特沃斯滤波后信号频谱（Welch 法）
plt.subplot(1, 2, 2)
plt.semilogy(f_welch_filtered, Pxx_welch_filtered)
plt.xlim(0, 50)
plt.title('Butterworth Filtered Signal Spectrum (Welch)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.show()