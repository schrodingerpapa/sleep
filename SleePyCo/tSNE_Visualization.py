import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.main_model import MainModel
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


config_path = r"/home/chenlungan/算法模型/SleePyCo/configs/SleePyCo-Transformer_SL-01_numScales-1_Sleep-EDF-2018_pretrain.json"
ckpt_path = r"/home/chenlungan/算法模型/SleePyCo/checkpoints/SleePyCo-Transformer_SL-01_numScales-1_Sleep-EDF-2018_pretrain/ckpt_fold-01.pth"
data_path = r"/home/chenlungan/公开数据集/Sleep-EDF-2018/npz/Fpz-Cz/SC4001E0.npz"

with open(config_path, 'r') as f:
    config = json.load(f)

data = np.load(data_path)
rawData = data['x']  # (N, T)
label = data['y']    # (N,)

print(f"Data shape: {rawData.shape}, Label unique: {np.unique(label)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MainModel(config).to(device)

# 加载 checkpoint
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(ckpt_path)

checkpoint = torch.load(ckpt_path, map_location=device)

# 处理可能的 key 前缀（如 'module.' 或 'feature.'）
state_dict = checkpoint
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']

# 移除可能的前缀（常见于 DDP 或 PL）
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('feature.'):
        new_state_dict[k[len('feature.'):]] = v
    elif k.startswith('module.feature.'):
        new_state_dict[k[len('module.feature.'):]] = v
    else:
        new_state_dict[k] = v

# 加载到 backbone
try:
    model.feature.load_state_dict(new_state_dict, strict=True)
except RuntimeError as e:
    print("⚠️ 严格加载失败，尝试非严格模式...")
    model.feature.load_state_dict(new_state_dict, strict=False)

model.eval()
print("✅ 模型加载成功！")

# ----------------------------
# 4. 提取特征（预训练模式 → c5）
# ----------------------------
def extract_features(model, data, device, batch_size=32):
    """
    提取预训练特征：对 c5 做全局平均池化 → (B, 256)
    """
    model.eval()
    features = []
    n_samples = data.shape[0]
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = data[i:i+batch_size]
            # 添加通道维度 (B, 1, T)
            x = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)
            
            # 获取 backbone 输出（列表，预训练时只有 c5）
            c5_list = model.feature(x)  # [c5]
            c5 = c5_list[0]  # (B, 256, 5)
            
            # 全局平均池化 → (B, 256)
            feat = torch.mean(c5, dim=2)  # 在时间维度上平均
            
            features.append(feat.cpu().numpy())
    
    return np.vstack(features)

# 转换数据为 float32
rawData = rawData.astype(np.float32)

# 提取特征
extracted_features = extract_features(model, rawData, device, batch_size=32)
print(f"Extracted features shape: {extracted_features.shape}")  # 应为 (N, 256)

# ----------------------------
# 5. t-SNE 可视化
# ----------------------------
# 标准化（强烈推荐）
scaler = StandardScaler()
features_scaled = scaler.fit_transform(extracted_features)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=42)
Data_2d = tsne.fit_transform(features_scaled)

# 绘图
plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'purple', 'red']
markers = ['*', 'o', '^', 'D', 'X']
sleep_stages = ['Wake', 'N1', 'N2', 'N3', 'REM']

for i in range(len(sleep_stages)):
    mask = label == i
    if np.any(mask):  # 确保该类别存在
        plt.scatter(Data_2d[mask, 0], Data_2d[mask, 1],
                    c=[colors[i]], marker=markers[i], s=30, alpha=0.8,
                    edgecolors='black', linewidth=0.5)

plt.title('t-SNE of Pretrained SleePyCo Features (c5 + GAP)', fontsize=16)
plt.axis('off')

# 图例
legend_elements = [
    plt.Line2D([0], [0], marker=mk, color='w', markerfacecolor=c, markersize=8, label=stage)
    for mk, c, stage in zip(markers, colors, sleep_stages)
]
plt.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.show()


