import numpy as np
import pandas as pd
import pywt
from antropy import sample_entropy, perm_entropy
from tqdm import tqdm # 显示进度条

def extract_features(epoch, fs):
    features = {}

    # 时域特征增强
    features['mean'] = np.mean(epoch)
    features['std'] = np.std(epoch)
    features['skewness'] = pd.Series(epoch).skew()
    features['kurtosis'] = pd.Series(epoch).kurtosis()
    features['zcr'] = (np.sum(np.abs(np.diff(np.sign(epoch)))) / 2) / (len(epoch) / fs)

    # Hjorth参数
    diff1 = np.diff(epoch)
    diff2 = np.diff(diff1)
    var0 = np.var(epoch)
    var1 = np.var(diff1)
    var2 = np.var(diff2)
    features['hjorth_activity'] = var0
    features['hjorth_mobility'] = np.sqrt(var1 / var0)
    features['hjorth_complexity'] = np.sqrt(var2 / var1) / features['hjorth_mobility']

    # 频域特征优化（调整频段范围）
    fft_vals = np.fft.rfft(epoch)
    psd = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(epoch), 1 / fs)

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'sigma': (12, 15),
        'beta': (15, 30),
        'gamma': (30, 40)
    }
    total_power = np.sum(psd)
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs < high)
        band_power = np.sum(psd[idx])
        features[f'{band}_power'] = band_power
        features[f'{band}_ratio'] = band_power / total_power  # 新增功率比

    # 小波变换改进（使用sym5小波）
    coeffs = pywt.wavedec(epoch, 'sym5', level=6)  # 增加分解层数
    for i, coeff in enumerate(coeffs):
        features[f'wavelet_energy_{i}'] = np.sum(coeff ** 2)
        features[f'wavelet_std_{i}'] = np.std(coeff)
        features[f'wavelet_entropy_{i}'] = perm_entropy(coeff, order=3, delay=1)

    # 非线性特征参数优化
    features['sample_entropy'] = sample_entropy(epoch, order=2, metric='chebyshev')
    features['permutation_entropy'] = perm_entropy(epoch, order=4, delay=2)  # 增大order

    return features

def extract_all_features(epochs, fs):
    features_list = []
    for epoch in tqdm(epochs):
        features = extract_features(epoch, fs)
        features_list.append(features)
    return pd.DataFrame(features_list)