import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend, periodogram
import matplotlib; matplotlib.use('TkAgg')


EEG_path0 = r"C:\算法复现\Sleep\zz_ks1092\fpz_pre_n1_wake_epochs_50_55.csv"

# 加载数据并定义参数
data = pd.read_csv(EEG_path0).values
x=data[:,5]
Fs = 500    # 采样率
N = len(x)               # 样本数
x = detrend(x)           # 去除趋势项


# 计算周期图
nfft = 2 ** int(np.ceil(np.log2(N)))  # 计算最近的2的幂次
f, P_per = periodogram(x, fs=Fs, window='boxcar', nfft=nfft, scaling='density')

# 提取0-50Hz范围内的频率
mask = (f > 0) & (f <= 70)
f_lim = f[mask]
P_lim = P_per[mask]

# 绘制结果
plt.figure(figsize=(12, 5))

# 线性坐标
plt.subplot(1, 2, 1)
plt.plot(f_lim, P_lim, 'k', linewidth=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (μV²/Hz)')
plt.title('Periodogram (Linear Scale)')
plt.xlim([f_lim[0], f_lim[-1]])

# 对数坐标
plt.subplot(1, 2, 2)
plt.plot(f_lim, 10 * np.log10(P_lim), 'k', linewidth=1)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (dB)')
plt.title('Periodogram (Logarithmic Scale)')
plt.xlim([f_lim[0], f_lim[-1]])

plt.tight_layout()
plt.show()