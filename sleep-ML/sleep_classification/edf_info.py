import mne
import matplotlib
matplotlib.use('TkAgg')  # 或者其他后端如
import matplotlib.pyplot as plt

# 读取EEG信号文件
eeg_file = r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"
raw = mne.io.read_raw_edf(eeg_file, preload=True)
# 读取Hypnogram注释文件
hypno_file = r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001EC-Hypnogram.edf"
annotations = mne.read_annotations(hypno_file)
print(annotations.onset)

raw.set_annotations(annotations)
print(raw.annotations)

# 绘制原始信号和注释
raw.plot(start=60, duration=120)  # 查看第1-3分钟的数据
plt.show(block=True)

