import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt, periodogram, welch, butter, sosfilt
matplotlib.use('TkAgg')  # 或者尝试 'QtAgg', 'TkAgg' 等其他后端

# 读取文件
file_path = r"C:\Users\clg\Desktop\测试--SW3011采集数据\50μV信号发生器--V3.csv"  # 替换为你的文件路径
data = pd.read_csv(file_path, header=None)  # 假设没有表头
data = data.iloc[:, ]  # 假设第一列是时间戳，我们忽略它
signal = data.values.flatten()  # 将数据转换为一维数组


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    y = sosfilt(sos, data)
    return y

lowcut = 0.5  # 低频截止频率
highcut = 35  # 高频截止频率
fs=500
signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order=4)

# 绘制原始信号
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(signal)
plt.title('Original Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# 中值滤波，窗口大小设置为 31（可以根据实际情况调整）
window_size = 21
filtered_signal = medfilt(signal, kernel_size=window_size)

# 绘制滤波后的信号
plt.subplot(2, 1, 2)
plt.plot(filtered_signal)
plt.title('Filtered Signal with Median Filter')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# 假设采样频率为 500 Hz（根据实际情况调整）
fs = 500  # 采样频率
xlim = (0, 100)  # 设置 x 轴的范围

# 使用周期图法计算频谱
f_periodogram, Pxx_periodogram = periodogram(signal, fs=fs)
f_periodogram_filtered, Pxx_periodogram_filtered = periodogram(filtered_signal, fs=fs)

# 使用 Welch 法计算频谱
f_welch, Pxx_welch = welch(signal, fs=fs, window='hann', nperseg=1024, noverlap=512)
f_welch_filtered, Pxx_welch_filtered = welch(filtered_signal, fs=fs, window='hann', nperseg=1024, noverlap=512)

# 绘制原始信号的频谱
plt.figure(figsize=(12, 8))

# 原始信号频谱（FFT）
plt.subplot(2, 2, 1)
n = len(signal)
freq = np.fft.fftfreq(n, 1/fs)
fft_spectrum_original = np.abs(np.fft.fft(signal)) / n
plt.plot(freq[:n//2], fft_spectrum_original[:n//2])
plt.xlim(0, 250)
plt.title('Original Signal Spectrum (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# 原始信号频谱（周期图法）
plt.subplot(2, 2, 2)
plt.semilogy(f_periodogram, Pxx_periodogram)
plt.xlim(0, 250)
plt.title('Original Signal Spectrum (Periodogram)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

# 滤波后信号频谱（FFT）
plt.subplot(2, 2, 3)
fft_spectrum_filtered = np.abs(np.fft.fft(filtered_signal)) / n
plt.plot(freq[:n//2], fft_spectrum_filtered[:n//2])
plt.xlim(0, 250)
plt.title('Filtered Signal Spectrum (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# 滤波后信号频谱（周期图法）
plt.subplot(2, 2, 4)
plt.semilogy(f_periodogram_filtered, Pxx_periodogram_filtered)
plt.xlim(0, 250)
plt.title('Filtered Signal Spectrum (Periodogram)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.show()

# 绘制原始信号和滤波后信号的 Welch 频谱
plt.figure(figsize=(12, 8))

# 原始信号频谱（Welch 法）
plt.subplot(1, 2, 1)
plt.semilogy(f_welch, Pxx_welch)
plt.xlim(0, 50)
plt.title('Original Signal Spectrum (Welch)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

# 滤波后信号频谱（Welch 法）
plt.subplot(1, 2, 2)
plt.semilogy(f_welch_filtered, Pxx_welch_filtered)
plt.xlim(0, 50)
plt.title('Filtered Signal Spectrum (Welch)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.show()

# 添加 0.5-40Hz 的 4 阶巴特沃斯滤波


# 绘制巴特沃斯滤波后的信号
plt.figure(figsize=(10, 4))
plt.plot(filtered_signal)
plt.title('Filtered Signal with Butterworth Filter (0.5-40Hz)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# 使用周期图法计算巴特沃斯滤波后的频谱
f_butterworth_periodogram, Pxx_butterworth_periodogram = periodogram(filtered_signal, fs=fs)

# 使用 Welch 法计算巴特沃斯滤波后的频谱
f_butterworth_welch, Pxx_butterworth_welch = welch(filtered_signal, fs=fs, window='hann', nperseg=1024, noverlap=512)

# 绘制巴特沃斯滤波后的频谱
plt.figure(figsize=(12, 8))

# 巴特沃斯滤波后信号频谱（FFT）
plt.subplot(2, 2, 1)
fft_spectrum_butterworth = np.abs(np.fft.fft(filtered_signal)) / n
plt.plot(freq[:n//2], fft_spectrum_butterworth[:n//2])
plt.xlim(0, 50)
plt.title('Butterworth Filtered Signal Spectrum (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# 巴特沃斯滤波后信号频谱（周期图法）
plt.subplot(2, 2, 2)
plt.semilogy(f_butterworth_periodogram, Pxx_butterworth_periodogram)
plt.xlim(0, 50)
plt.title('Butterworth Filtered Signal Spectrum (Periodogram)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

# 巴特沃斯滤波后信号频谱（Welch 法）
plt.subplot(2, 2, 3)
plt.semilogy(f_butterworth_welch, Pxx_butterworth_welch)
plt.xlim(0, 50)
plt.title('Butterworth Filtered Signal Spectrum (Welch)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')

plt.tight_layout()
plt.show()