import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# 定义睡眠阶段映射
stage_mapping = {
    'Sleep stage W': 'Wake',
    'Sleep stage 1': 'N1',
    'Sleep stage 2': 'N2',
    'Sleep stage 3': 'N3',
    'Sleep stage 4': 'N3',
    'Sleep stage R': 'REM',
    'Sleep stage ?': 'Unlabeled'
}

start_epoch=20,
end_epoch=25

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def find_first_n1_onset(annotations):
    """找到第一个N1阶段的开始时间"""
    for ann in annotations:
        if stage_mapping.get(ann['description'], '') == 'N1':
            return ann['onset']
    return None

def extract_specific_wake_epochs(eeg_data, fs, annotations, epoch_length=30, start_epoch=20, end_epoch=25):
    """提取N1阶段前指定范围的Wake期数据"""
    first_n1_onset = find_first_n1_onset(annotations)

    if first_n1_onset is None:
        print("未找到N1阶段，无法提取数据")
        return None

    wake_candidates = []
    for ann in annotations:
        desc = ann['description']
        if stage_mapping.get(desc, '') != 'Wake':
            continue

        ann_start = ann['onset']
        ann_end = ann['onset'] + ann['duration']

        if ann_end <= first_n1_onset:
            wake_candidates.append((ann_start, ann_end))

    wake_candidates.sort(key=lambda x: x[1])  # 按结束时间升序排列

    collected_epochs = []
    for start, end in wake_candidates:
        max_epochs = int((end - start) // epoch_length)

        for i in range(max_epochs):
            epoch_end = start + (i + 1) * epoch_length
            epoch_start = start + i * epoch_length

            if epoch_end > first_n1_onset:
                break

            start_idx = int(epoch_start * fs)
            end_idx = int(epoch_end * fs)

            if start_idx < 0 or end_idx > len(eeg_data):
                continue

            epoch = eeg_data[start_idx:end_idx]
            collected_epochs.append(epoch)

    if len(collected_epochs) < end_epoch:
        print(f"警告：仅找到{len(collected_epochs)}个Wake期")
        return None

    return collected_epochs[start_epoch-1:end_epoch]  # 返回第50到第55个epoch

def save_epochs_to_csv(epochs, output_path):
    if not epochs:
        print("未找到有效数据，无法保存")
        return

    min_length = min(len(epoch) for epoch in epochs)
    trimmed = [epoch[:min_length] for epoch in epochs]

    df = pd.DataFrame({
        f'Wake_{i + 1}': epoch
        for i, epoch in enumerate(trimmed)
    })

    df.to_csv(output_path, index=False)
    print(f"数据已保存至：{output_path}")

if __name__ == "__main__":
    eeg_file = r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf"
    annotation_file = r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001EC-Hypnogram.edf"

    raw = mne.io.read_raw_edf(eeg_file, preload=True)
    channel_names = raw.info['ch_names']

    fpz_index = None
    for i, ch_name in enumerate(channel_names):
        if 'FPZ' in ch_name.upper():
            fpz_index = i
            break

    if fpz_index is None:
        print("未找到FPz电极，请检查数据中的通道名称")
        exit()

    eeg_data = raw.get_data(picks=fpz_index)[0]
    fs = int(raw.info['sfreq'])
    annotations = mne.read_annotations(annotation_file)

    eeg_data = butter_bandpass_filter(eeg_data, 0.5, 35, fs)

    target_epochs = extract_specific_wake_epochs(
        eeg_data=eeg_data,
        fs=fs,
        annotations=annotations,
        start_epoch=20,
        end_epoch=25
    )

    if target_epochs:
        save_epochs_to_csv(
            epochs=target_epochs,
            output_path="../../zEEG-test/zz_ks1092/fpz_pre_n1_wake_epochs_20_25.csv"
        )