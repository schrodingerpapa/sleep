import pandas as pd
import matplotlib.pyplot as plt  # 可视化库，用于显示结果
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import numpy as np
from PyEMD import EMD, Visualisation
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 指定常用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

############私人函数定义区###################################

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


def show_freq(data, Fs=1000):
    L = len(data)
    T = 1.0 / Fs  # 采样周期 (s)

    yf = fft(data)
    yf = 2 * np.abs(yf[:(L//2)]/L)  # 取FFT结果的绝对值
    yf = (yf/np.max(yf))**2

    xf = np.arange(0, (L//2)) * (Fs/L)

    return xf, yf


# def My_plot_freq(data, xf, yf):
#     L = len(data)
#     # 绘制时序信号
#     plt.figure(figsize=(12, 6))
#
#     plt.plot(xf, 2.0 / L * yf)  # 2.0/L 是为了缩放FFT结果，以便显示正确的幅度
#     plt.title('Frequency Spectrum')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Amplitude')
#
#     plt.tight_layout()
#     plt.show()

def My_plot_freq(xf, yf):
    # 绘制时序信号
    plt.figure(figsize=(6, 6))

    plt.plot(xf, yf)  # 2.0/L 是为了缩放FFT结果，以便显示正确的幅度
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

def My_plot_freq1(xf1, yf1,xf2, yf2,xf3, yf3):
    # 绘制时序信号
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.plot(xf1, yf1)
    plt.title('brain product——FP1')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 2)
    plt.plot(xf2, yf2)
    plt.title('brain product——FP2')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 3)
    plt.plot(xf3, yf3)
    plt.title('设备一通道')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')



    plt.tight_layout()
    plt.show()


def My_plot_freq2(data, t, xf, yf):
    L = len(data)
    # 绘制时序信号
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, data)
    plt.title('Time-Series Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # 绘制频谱
    plt.subplot(2, 1, 2)
    # plt.plot(xf, 2.0 / L * yf)  # 2.0/L 是为了缩放FFT结果，以便显示正确的幅度
    plt.plot(xf, yf)  # 2.0/L 是为了缩放FFT结果，以便显示正确的幅度
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
    plt.title('brain product——FP1')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 2)
    plt.plot(t2, data2)
    plt.title('brain product——FP2')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 2, 3)
    plt.plot(t3, data3)
    plt.title('设备一通道')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':



    # 文件路径
    father_path = r"C:\研究生\脑电实验测试数据以及相关代码\设备实验汇总\3-16专业设备心算任务\心算任务测试1"
    EEG_path0 = father_path+r"\Fp1_RawData.csv"
    EEG_path1 = father_path+r"\Fp2_RawData.csv"
    EEG_path2 = father_path+r"\Fp2_RawData.csv"


    # 读取CSV文件
    EEG_df0 = pd.read_csv(EEG_path0).values
    EEG_df1 = pd.read_csv(EEG_path1).values
    EEG_df2 = pd.read_csv(EEG_path2).values




    #显示窗口
    N = EEG_df0.shape[0]
    print(N)
    test_long = N
    start_long = 0

    t_0 = range(start_long, start_long + test_long)
    print(t_0)
    t_1 = range(start_long, start_long + test_long)
    t_2 = range(start_long, start_long + test_long)

    data_tmp0 = EEG_df0[start_long:start_long+test_long, 1]
    data_tmp1 = EEG_df1[start_long:start_long+test_long, 1]
    data_tmp2 = EEG_df2[start_long:start_long+test_long, 1]


    # 数据预处理区
    Fs = 1000
    lowcut = 1  # 低截止频率，单位为Hz
    highcut = 40  # 高截止频率，单位为Hz

    xf0, yf0 = show_freq(EEG_df0[:, 1], Fs=1000)
    xf1, yf1 = show_freq(EEG_df1[:, 1], Fs=1000)
    xf2, yf2 = show_freq(EEG_df2[:, 1], Fs=1000)
    freq_test =N
    #绘制原始数据频谱图
    My_plot_freq1(xf0[:freq_test], yf0[:freq_test], xf1[:freq_test], yf1[:freq_test], xf2[:freq_test], yf2[:freq_test])
    #绘制原始数据图
    My_plot2(t_0, data_tmp0, t_1, data_tmp1, t_2, data_tmp2)

    # 对滤波后的数据进行陷波滤波
    filtered_wave_0 = notch_filter(EEG_df0[start_long:start_long+test_long, 1], Fs)
    filtered_wave_1 = notch_filter(EEG_df0[start_long:start_long+test_long, 1], Fs)
    filtered_wave_2 = notch_filter(EEG_df0[start_long:start_long+test_long, 1], Fs)

    # 对原始数据进行带通滤波
    filtered_wave_0 = butter_bandpass_filter(filtered_wave_0, lowcut, highcut, Fs, order=4)
    filtered_wave_1 = butter_bandpass_filter(filtered_wave_1, lowcut, highcut, Fs, order=4)
    filtered_wave_2 = butter_bandpass_filter(filtered_wave_2, lowcut, highcut, Fs, order=4)


    #绘制滤波后的时序图
    My_plot2(t_0, filtered_wave_0, t_1, filtered_wave_1, t_2, filtered_wave_2)

    #绘制滤波后的频谱图
    f_l = N
    xf10, yf10 = show_freq(filtered_wave_0, Fs=1000)
    xf11, yf11 = show_freq(filtered_wave_1, Fs=1000)
    xf12, yf12 = show_freq(filtered_wave_2, Fs=1000)
    My_plot_freq1(xf10[:f_l], yf10[:f_l], xf11[:f_l], yf11[:f_l], xf12[:f_l], yf12[:f_l])


