import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import matplotlib
matplotlib.use('TkAgg')  # Add this before importing pyplot
import matplotlib .pyplot as plt
from extractFeature import extract_all_features
from preprocessing import load_edf_data,segment_epochs

def load_and_process_participant(eeg_file, annotation_file):
    """加载单个参与者的数据并提取特征"""
    eeg_data, fs, annotations = load_edf_data(eeg_file, annotation_file)
    epochs, labels = segment_epochs(eeg_data, fs, annotations)
    feature_df = extract_all_features(epochs, fs)
    feature_df['label'] = labels
    feature_df = feature_df[feature_df['label'] != 'Unlabeled']
    return feature_df

# 修改后的睡眠分期可视化代码
def plot_hypnogram(true_labels, pred_labels, fs=100, epoch_length=30):
    # 创建阶段到数值的映射
    stage_order = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}
    colors = {'Wake': '#FF6961', 'N1': '#77B5FE', 'N2': '#0165FC', 'N3': '#000072', 'REM': '#03BB92'}

    # 转换为数值和颜色序列
    true_numeric = [stage_order[label] for label in true_labels]
    pred_numeric = [stage_order[label] for label in pred_labels]
    true_colors = [colors[label] for label in true_labels]
    pred_colors = [colors[label] for label in pred_labels]

    # 创建时间轴（单位：小时）
    total_epochs = len(true_labels)
    time_hours = np.arange(total_epochs) * epoch_length / 3600

    # 创建画布
    plt.figure(figsize=(15, 8))

    # 绘制真实Hypnogram
    plt.subplot(2, 1, 1)
    plt.step(time_hours, true_numeric, where='post', color='black', linewidth=0.8)
    plt.fill_between(time_hours, true_numeric, step='post',
                     color=true_colors, alpha=0.4)
    plt.title('True Hypnogram', fontsize=12, pad=10)
    plt.ylabel('Sleep Stage', fontsize=10)
    plt.yticks([0, 1, 2, 3, 4], ['Wake', 'N1', 'N2', 'N3', 'REM'])
    plt.ylim(-0.5, 4.5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    # 绘制预测Hypnogram
    plt.subplot(2, 1, 2)
    plt.step(time_hours, pred_numeric, where='post', color='black', linewidth=0.8)
    plt.fill_between(time_hours, pred_numeric, step='post',
                     color=pred_colors, alpha=0.4)
    plt.title(f'Predicted Hypnogram (Accuracy: {accuracy_score(y_test, y_pred):.2%})',
              fontsize=12, pad=10)
    plt.ylabel('Sleep Stage', fontsize=10)
    plt.xlabel('Time (hours)', fontsize=10)
    plt.yticks([0, 1, 2, 3, 4], ['Wake', 'N1', 'N2', 'N3', 'REM'])
    plt.ylim(-0.5, 4.5)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 定义参与者的文件路径（前五人为训练，第六人测试）
    participants = [
        {'eeg': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001E0-PSG.edf",
         'annot': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4001EC-Hypnogram.edf"},
        {'eeg': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4002E0-PSG.edf",
         'annot': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4002EC-Hypnogram.edf"},
        {'eeg': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4011E0-PSG.edf",
         'annot': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4011EH-Hypnogram.edf"},
        {'eeg': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012E0-PSG.edf",
         'annot': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4012EC-Hypnogram.edf"},
        {'eeg': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4021E0-PSG.edf",
         'annot': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4021EH-Hypnogram.edf"},
    ]

    test_participant = {
        'eeg': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4022E0-PSG.edf",
        'annot': r"C:\公开数据集\睡眠脑电公开数据集\sleep-edf-database-expanded-1.0.0\sleep-cassette\SC4022EJ-Hypnogram.edf"
    }

    # 加载并处理前五人数据
    train_data = []
    for participant in participants:
        feature_df = load_and_process_participant(participant['eeg'], participant['annot'])
        train_data.append(feature_df)
    train_df = pd.concat(train_data, ignore_index=True)

    # 加载并处理测试人数据
    test_df = load_and_process_participant(test_participant['eeg'], test_participant['annot'])

    # 分离特征和标签
    X_train = train_df.drop('label', axis=1)
    X_train = X_train.fillna(X_train.mean())
    y_train = train_df['label']

    X_test = test_df.drop('label', axis=1)
    X_test = X_test.fillna(X_train.mean())
    y_test = test_df['label']

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练随机森林模型
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf.fit(X_train_scaled, y_train)

    # 保存训练的模型
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(clf, 'sleep_rf_model.pkl')

    # 评估
    y_pred = clf.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Cohen's Kappa:", cohen_kappa_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # 特征重要性分析
    importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("Top 10 Features:\n", importances.head(10))

    # 调用修改后的可视化函数
    plot_hypnogram(y_test.values, y_pred)