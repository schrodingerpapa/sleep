import numpy as np
import pandas as pd
from extractFeature import extract_all_features
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\算法复现\Sleep\sleep-ML\processed_eeg_data.csv")
eeg_data = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().values
fs = 100
epoch_length = 30  # 秒
epoch_len = fs * epoch_length

n_epochs = len(eeg_data) // epoch_len
epochs = eeg_data[:n_epochs * epoch_len].reshape(n_epochs, epoch_len)

def plot_private_hypnogram(pred_labels, Fs=fs, epoch_length=30):
    stage_order = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
    colors = {'Wake': '#FF6961', 'N1': '#77B5FE', 'N2': '#0165FC', 'N3': '#000072', 'REM': '#03BB92'}

    pred_numeric = [stage_order.get(label, -1) for label in pred_labels]
    pred_colors = [colors.get(label, '#CCCCCC') for label in pred_labels]
    total_epochs = len(pred_labels)
    time_hours = np.arange(total_epochs) * epoch_length / 3600

    plt.figure(figsize=(12, 4))
    plt.step(time_hours, pred_numeric, where='post', color='black', linewidth=0.8)
    plt.fill_between(time_hours, pred_numeric, step='post', color=pred_colors, alpha=0.4)
    plt.title('Predicted Hypnogram for Private Data')
    plt.ylabel('Sleep Stage')
    plt.xlabel('Time (hours)')
    plt.yticks([0, 1, 2, 3, 4], ['Wake', 'N1', 'N2', 'N3', 'REM'])
    plt.ylim(-0.5, 4.5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# 批量提取特征，返回DataFrame
private_features = extract_all_features(epochs, fs)

# 替换 inf/-inf 为 NaN，并用均值填充
private_features = private_features.replace([np.inf, -np.inf], np.nan)
private_features = private_features.fillna(private_features.mean())

# 加载训练时保存的scaler
scaler = joblib.load(r"C:\算法复现\Sleep\sleep-ML\sleep_classification\scaler.pkl")
private_scaled = scaler.transform(private_features)
clf = joblib.load(r"C:\算法复现\Sleep\sleep-ML\sleep_classification\sleep_rf_model.pkl")
private_pred = clf.predict(private_scaled)

# 保存或输出预测标签
pd.DataFrame({'predicted_label': private_pred}).to_csv('private_pred.csv', index=False)
print(private_pred)

# 调用可视化
plot_private_hypnogram(private_pred)
