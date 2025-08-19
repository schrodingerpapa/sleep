import pandas as pd
import matplotlib.pyplot as plt  # 可视化库，用于显示结果
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import numpy as np



############私人函数定义区###################################
def high_V_filter(data, V):
    for n in range(10):
        tmp_data = data[1000 * n:1000 * n + 1000]
        data_med = np.median(np.abs(tmp_data))
        tmp_data = data_med + tmp_data
        data_med = np.median(np.abs(tmp_data))
        FVH = data_med + V
        FVL = data_med - V
        for i in range(tmp_data.shape[0]):
            if (np.abs(tmp_data[i]) > FVH) | (np.abs(tmp_data[i]) < FVL):
                tmp_data[i] = data_med
        data[1000 * n:1000 * n + 1000] = tmp_data
    return data


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


# 应用滤波器
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


from scipy.signal import iirnotch, filtfilt


# 设计50Hz的-notch滤波器
def notch_filter(data, fs, Q=30):
    f0 = 50.0  # 工频干扰频率
    b, a = iirnotch(f0, Q, fs)
    y = filtfilt(b, a, data)
    return y








# 生成示例时序数据（例如，一个包含多个频率成分的信号）
def general_test_signal(Fs=1000, L=1500):
    T = 1.0 / Fs  # 采样周期 (s)
    t = np.linspace(0.0, L * T, L, endpoint=False)  # 时间向量

    # 创建一个包含50 Hz和120 Hz信号的合成信号
    signal = 0.7 * np.sin(2 * np.pi * 50.0 * t) + np.sin(2 * np.pi * 120.0 * t)

    # 添加一些噪声
    noise = 0.5 * np.random.randn(L)
    signal += noise
    return signal, t


def show_freq(data, Fs=1000):
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


def My_plot_freq1(xf1, yf1, xf2, yf2, xf3, yf3):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.plot(xf1, yf1)
    plt.title('Ch1 Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 2)
    plt.plot(xf2, yf2)
    plt.title('Ch2 Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 3)
    plt.plot(xf3, yf3)
    plt.title('Ch3 Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def My_plot_freq2(data, t, xf, yf):
    L = len(data)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, data)
    plt.title('Time-Series Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(xf, yf)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def My_plot(t1, data1):
    plt.figure(figsize=(12, 12))
    plt.plot(t1, data1)
    plt.title('Test Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


def My_plot2(t1, data1, t2, data2, t3, data3):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.plot(t1, data1)
    plt.title('Ch1 Time-Series Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 2)
    plt.plot(t2, data2)
    plt.title('Ch2 Time-Series Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 3)
    plt.plot(t3, data3)
    plt.title('Ch3 Time-Series Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def replace_outliers_with_nearest(data, upper_threshold, lower_threshold):
    """
    将超过阈值的数据点用最近的未超过阈值的数据点填充。
    :param data: 输入数据数组
    :param upper_threshold: 上限阈值
    :param lower_threshold: 下限阈值
    :return: 去除尖峰后的数据数组
    """
    # 将数据转换为 Pandas Series
    series = pd.Series(data)

    # 将超过阈值的点标记为 NaN
    series[(series > upper_threshold) | (series < lower_threshold)] = np.nan

    # 使用线性插值填充 NaN
    series.interpolate(method='linear', limit_direction='both', inplace=True)

    # 将结果转换回 NumPy 数组
    return series.values


if __name__ == '__main__':
    # 文件路径
    father_path = r"C:\Users\clg\Desktop\25-05-22-10-08-16"
    EEG_path0 = father_path + r"\EEG_000003_00.csv"
    EEG_path1 = father_path + r"\EEG_000003_01.csv"
    EEG_path2 = father_path + r"\EEG_000003_02.csv"

    # 读取CSV文件
    EEG_df0 = pd.read_csv(EEG_path0).values
    EEG_df1 = pd.read_csv(EEG_path1).values
    EEG_df2 = pd.read_csv(EEG_path2).values

    # 显示窗口
    N = EEG_df0.shape[0]
    print(f"Total data length: {N}")

    test_long =20000
    start_long = 6000
    t1 = range(start_long, start_long + test_long)

    # 提取三个通道数据
    data_tmp0 = EEG_df0[start_long:start_long + test_long, 1]
    data_tmp1 = EEG_df1[start_long:start_long + test_long, 1]
    data_tmp2 = EEG_df2[start_long:start_long + test_long, 1]

    # # 替换异常值为最近的有效值
    # data_tmp0 = replace_outliers_with_nearest(data_tmp0, upper_threshold=-151000, lower_threshold=-180000)
    # data_tmp1 = replace_outliers_with_nearest(data_tmp1, upper_threshold=-171000, lower_threshold=-172240)
    # data_tmp2 = replace_outliers_with_nearest(data_tmp2, upper_threshold=-171000, lower_threshold=-172240)
    from scipy.signal import medfilt
    data_tmp0 = medfilt(data_tmp0, kernel_size=3)
    data_tmp1 = medfilt(data_tmp1, kernel_size=3)
    data_tmp2 = medfilt(data_tmp2, kernel_size=3)
    # 数据统计信息
    print("Mean values after cleaning:",
          np.mean(data_tmp0), np.mean(data_tmp1), np.mean(data_tmp2))
    print("Std values after cleaning:",
          np.std(data_tmp0), np.std(data_tmp1), np.std(data_tmp2))

    # 数据预处理
    Fs = 200
    lowcut = 1  # 低截止频率，单位为Hz
    highcut = 40  # 高截止频率，单位为Hz

    # 显示原始数据的频谱
    xf0, yf0 = show_freq(data_tmp0, Fs=1000)
    xf1, yf1 = show_freq(data_tmp1, Fs=1000)
    xf2, yf2 = show_freq(data_tmp2, Fs=1000)
    freq_test = 50
    My_plot_freq1(xf0[:freq_test], yf0[:freq_test],
                  xf1[:freq_test], yf1[:freq_test],
                  xf2[:freq_test], yf2[:freq_test])

    # 绘制原始数据的时间序列
    t_0 = range(start_long, start_long + test_long)
    t_1 = range(start_long, start_long + test_long)
    t_2 = range(start_long, start_long + test_long)
    My_plot2(t_0, data_tmp0, t_1, data_tmp1, t_2, data_tmp2)

    # 带通滤波
    filtered_wave_0 = butter_bandpass_filter(data_tmp0, lowcut, highcut, Fs, order=5)
    filtered_wave_1 = butter_bandpass_filter(data_tmp1, lowcut, highcut, Fs, order=5)
    filtered_wave_2 = butter_bandpass_filter(data_tmp2, lowcut, highcut, Fs, order=5)

    # notch滤波
    filtered_wave_0 = notch_filter(filtered_wave_0, 150)
    filtered_wave_1 = notch_filter(filtered_wave_1, 150)
    filtered_wave_2 = notch_filter(filtered_wave_2, 150)



    # 绘制处理后的数据频谱
    f_l = N
    xf10, yf10 = show_freq(filtered_wave_0, Fs=1000)
    xf11, yf11 = show_freq(filtered_wave_1, Fs=1000)
    xf12, yf12 = show_freq(filtered_wave_2, Fs=1000)
    My_plot_freq1(xf10[:f_l], yf10[:f_l],
                  xf11[:f_l], yf11[:f_l],
                  xf12[:f_l], yf12[:f_l])