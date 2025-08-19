import scipy.signal
import numpy as np

def butter_bandpass(data,lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered_data = scipy.signal.filtfilt(b, a, data)
    return filtered_data

def butter_notch(data,notch_freq, fs, quality_factor=30):
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = scipy.signal.butter(2, [freq - 0.1 / quality_factor, freq + 0.1 / quality_factor], btype='bandstop')
    filtered_data = scipy.signal.filtfilt(b, a, data)
    return filtered_data

def median_filter(signal, window_size=21):
    """中值滤波"""
    if window_size % 2 == 0:
        raise ValueError("窗口大小必须为奇数")
    return scipy.signal.medfilt(signal, kernel_size=window_size)

def mean_filter(signal, window_size=21):
    """均值滤波"""
    if window_size % 2 == 0:
        raise ValueError("窗口大小必须为奇数")
    return scipy.signal.convolve(signal, np.ones(window_size)/window_size, mode='same')

def SG_filter(signal, window_size=21, polyorder=2):
    """Savitzky-Golay滤波"""
    if window_size % 2 == 0:
        raise ValueError("窗口大小必须为奇数")
    return scipy.signal.savgol_filter(signal, window_size, polyorder)

# todo：自适应SG滤波
# 自适应逻辑




