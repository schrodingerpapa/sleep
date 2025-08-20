import mne
import numpy as np
from scipy.signal import butter, filtfilt

# 定义睡眠阶段映射（R&K标准 → AASM标准）
stage_mapping = {
    'Sleep stage W': 'Wake',
    'Sleep stage 1': 'N1',
    'Sleep stage 2': 'N2',
    'Sleep stage 3': 'N3',
    'Sleep stage 4': 'N3',  # 合并N3和N4为深睡期
    'Sleep stage R': 'REM',
    'Sleep stage ?': 'Unlabeled'
}

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def load_edf_data(eeg_file_path, annotation_file_path):
    """加载EDF文件和注释文件"""
    eeg_raw = mne.io.read_raw_edf(eeg_file_path, preload=True)
    eeg_data = eeg_raw.get_data(picks='eeg')[0]
    fs = int(eeg_raw.info['sfreq'])  # 采样率（通常为100 Hz）
    eeg_data = butter_bandpass_filter(eeg_data, 0.3, 35, fs)  # AASM中EEG滤波推荐0.3-35hz
    annotations = mne.read_annotations(annotation_file_path)

    return eeg_data, fs, annotations
def segment_epochs(eeg_data, fs, annotations, epoch_length=30):
    """动态分割为固定30秒的epoch，包含真实睡眠分期及睡眠前100帧清醒数据"""
    epochs, labels = [], []
    # 步骤1: 确定睡眠期的开始和结束时间（第一个非Wake到最后一个非Wake阶段），筛选所有睡眠期
    sleep_stage_annots = []
    for ann in annotations:
        if ann['description'] not in stage_mapping: continue
        label = stage_mapping[ann['description']]
        if label not in ['Wake', 'Unlabeled']:
            sleep_stage_annots.append(ann)

    if not sleep_stage_annots: # 如果没有睡眠期，则返回空列表，不执行后续操作
        return np.array(epochs), np.array(labels)

    sleep_start = min(ann['onset'] for ann in sleep_stage_annots)
    sleep_end = max(ann['onset'] + ann['duration'] for ann in sleep_stage_annots)

    # 步骤2: 收集睡眠开始前100帧清醒数据
    pre_wake_epochs, pre_wake_labels = [], []
    wake_candidates = []

    # 提取所有清醒期候选片段
    for ann in annotations:
        if ann['description'] not in stage_mapping: continue
        mapped_label = stage_mapping[ann['description']]
        if mapped_label != 'Wake': continue

        ann_start = ann['onset']
        ann_end = ann['onset'] + ann['duration']

        # 只处理睡眠开始前的清醒期
        if ann_end <= sleep_start:
            wake_candidates.append((ann_start, ann_end))

    # 从近到远选择时间片段
    wake_candidates.sort(reverse=True, key=lambda x: x[1])
    required_frames = 100

    for start, end in wake_candidates:
        available_epochs = int((end - start) // epoch_length)
        available_epochs = min(available_epochs, required_frames)

        for i in range(available_epochs):
            epoch_start = end - (i + 1) * epoch_length
            epoch_end = epoch_start + epoch_length

            start_idx = int(epoch_start * fs)
            end_idx = int(epoch_end * fs)

            if end_idx > len(eeg_data): continue

            epoch = eeg_data[start_idx:end_idx]
            pre_wake_epochs.insert(0, epoch)  # 保持时间顺序
            pre_wake_labels.insert(0, 'Wake')

        required_frames -= available_epochs
        if required_frames <= 0: break

    # 步骤3: 处理真实睡眠分期数据（原逻辑）
    sleep_epochs, sleep_labels = [], []

    for ann in annotations:
        if ann['description'] not in stage_mapping: continue
        mapped_label = stage_mapping[ann['description']]
        if mapped_label == 'Unlabeled': continue

        # 当前注释的时间范围
        ann_onset = ann['onset']
        ann_duration = ann['duration']
        ann_end = ann_onset + ann_duration

        # 计算与睡眠期的重叠部分
        overlap_start = max(ann_onset, sleep_start)
        overlap_end = min(ann_end, sleep_end)
        if overlap_start >= overlap_end: continue

        # 计算可生成的epoch数量
        total_epochs = int((overlap_end - overlap_start) // epoch_length)

        for i in range(total_epochs):
            epoch_global_start = overlap_start + i * epoch_length
            epoch_global_end = epoch_global_start + epoch_length

            start_idx = int(epoch_global_start * fs)
            end_idx = int(epoch_global_end * fs)

            if end_idx > len(eeg_data): break

            epoch = eeg_data[start_idx:end_idx]
            sleep_epochs.append(epoch)
            sleep_labels.append(mapped_label)

    # 合并数据（前清醒+睡眠期）
    epochs = pre_wake_epochs + sleep_epochs
    labels = pre_wake_labels + sleep_labels

    return np.array(epochs), np.array(labels)