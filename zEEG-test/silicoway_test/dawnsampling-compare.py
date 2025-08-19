from scipy.signal import resample,resample_poly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import filter_processing

import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

print(filter_processing.__file__)


sampling_rate = 500 # 采样率
sampling_rate_target = 100  # 目标采样率
VREF = 2.42  # 参考电压
PGA = 12  # 可编程增益放大器增益
gain = 375
File_Path = r"C:\Users\clg\Desktop\测试--SW3011采集数据\午休_20250813_130347.csv"  # 替换为你的文件路径

def load_eeg_data(file_path):
    raw_data = pd.read_csv(file_path)
    """加载单通道EEG数据，从最后一列的第二行开始，支持csv和xlsx"""
    try:
        eeg_data = raw_data.iloc[1:, -1]  # 读取最后一列，从第二行开始
        #eeg = eeg_data*1e6*VREF/(gain*PGA*(2**23))
        eeg = eeg_data # 将数据转换为微
        return eeg.values
    except Exception as e:  # 捕获所有异常
        print(f"加载数据失败: {e}")
        return None

def resample_eeg(eeg_data, target_fs, original_fs=sampling_rate):  # 修改变量名为小写
    """重采样EEG数据到目标采样率"""
    if target_fs <= 0 or original_fs <= 0:
        raise ValueError("采样率必须大于0")

    # 计算重采样因子
    resample_factor = target_fs / original_fs

    # 使用resample_poly进行多项式重采样
    resampled_data = resample_poly(eeg_data, up=int(target_fs), down=int(original_fs))

    return resampled_data


eeg = load_eeg_data(File_Path)  # 替换为实际的文件路径
raw_time = np.arange(len(eeg)) / sampling_rate  # 原始时间轴
cut_time = 2 # 截取时间点（秒）
raw_eeg = eeg[raw_time >= cut_time]  # 截取cut_time后的EEG数据
raw_time = raw_time[raw_time >= cut_time]  # 截取120秒后的时间轴

resampled_eeg = resample_eeg(raw_eeg, sampling_rate_target)  # 重采样EEG数据
resampled_time = np.arange(len(resampled_eeg)) / sampling_rate_target  # 重采样后的时间轴

filtered_eeg = filter_processing.butter_bandpass(raw_eeg, 0.5, 40, sampling_rate)  # 带通滤波
filtered_eeg = filter_processing.butter_notch(filtered_eeg, 50, sampling_rate)  # 陷波滤波
# filtered_eeg = filter_processing.mean_filter(filtered_eeg, window_size=21)  # 均值滤波
# filtered_eeg = filter_processing.median_filter(filtered_eeg, window_size=21)
# filtered_eeg = filter_processing.SG_filter(filtered_eeg)  # 去除毛刺
# filtered_eeg = resample_eeg(filtered_eeg, sampling_rate)
resampled_eeg = resample_eeg(filtered_eeg, sampling_rate_target)  # 重采样EEG数据
# pd.save_csv = pd.DataFrame({'Time': resampled_time, 'EEG': filtered_eeg})
# 保存处理后的数据
# pd.save_csv.to_csv('processed_eeg_data.csv', index=False)

# 保存重采样后的数据
filename = '睡眠2-100hz'
save_path = r"C:\Users\clg\Desktop\模型和数据\{filename}.csv".format(filename=filename)
resampled_data_df = pd.DataFrame({'Time': resampled_time, 'EEG': resampled_eeg})
resampled_data_df.to_csv(save_path, index=False)


# 绘制原始信号
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(raw_time, raw_eeg, label='Original Signal', color='blue')
plt.title('Original EEG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV)')
# 绘制重采样后的信号
plt.subplot(2, 1, 2)
plt.plot(resampled_time, resampled_eeg, label='Resampled Signal', color='orange')
plt.title('Resampled EEG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV)')
plt.tight_layout()
plt.show()

time_filtered = np.arange(len(filtered_eeg)) / sampling_rate_target  # 截取后的时间轴
# 绘制截取后的信号
plt.figure(figsize=(10, 4))
plt.plot(time_filtered, filtered_eeg, label='Cut Signal', color='green')
plt.title('Cut EEG Signal from 120s')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV)')
plt.tight_layout()
plt.show()









