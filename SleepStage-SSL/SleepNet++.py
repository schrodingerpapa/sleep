import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing_ssl import UnsupervisedEEGDataset
import numpy as np

# 定义对比学习模型
class ContrastiveSleepStagingModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ContrastiveSleepStagingModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (input_size // 8 - 3), hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x), x  # 返回分类结果和特征表示


# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


# 自定义的 collate_fn
def custom_collate_fn(batch):
    # 获取批次中每个样本的两个增强视图
    aug1_list = [item[0] for item in batch]
    aug2_list = [item[1] for item in batch]

    # 确保所有样本长度一致
    max_len = max([len(aug1) for aug1 in aug1_list] + [len(aug2) for aug2 in aug2_list])

    # 对每个样本进行填充或截断
    aug1_padded = []
    aug2_padded = []
    for aug1, aug2 in zip(aug1_list, aug2_list):
        if len(aug1) < max_len:
            aug1 = np.pad(aug1, (0, max_len - len(aug1)), 'constant')
        else:
            aug1 = aug1[:max_len]
        aug1_padded.append(aug1)

        if len(aug2) < max_len:
            aug2 = np.pad(aug2, (0, max_len - len(aug2)), 'constant')
        else:
            aug2 = aug2[:max_len]
        aug2_padded.append(aug2)

    # 将列表转换为张量
    aug1_tensor = torch.FloatTensor(aug1_padded)
    aug2_tensor = torch.FloatTensor(aug2_padded)

    return aug1_tensor, aug2_tensor


# 设置超参数
input_size = 30 * 100  # 窗口长度30秒，采样率100Hz
hidden_size = 128
num_classes = 5  # 假设有5个睡眠阶段
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# 创建数据集和数据加载器
dataset = UnsupervisedEEGDataset("C:/公开数据集/睡眠脑电公开数据集/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4001E0-PSG.edf")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContrastiveSleepStagingModel(input_size, hidden_size, num_classes).to(device)
contrastive_criterion = ContrastiveLoss()
classification_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 生成固定的随机标签用于分类损失
fixed_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

# 训练模型
for epoch in range(num_epochs):
    for batch_idx, (aug1, aug2) in enumerate(dataloader):
        aug1 = aug1.unsqueeze(1).to(device)  # 添加通道维度
        aug2 = aug2.unsqueeze(1).to(device)

        # 前向传播
        output1, feature1 = model(aug1)
        output2, feature2 = model(aug2)

        # 对比损失
        contrastive_loss = contrastive_criterion(feature1, feature2, torch.ones(batch_size).to(device))

        # 分类损失使用固定标签
        classification_loss = classification_criterion(output1, fixed_labels[:output1.size(0)])

        # 总损失函数
        total_loss = contrastive_loss + classification_loss

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'contrastive_sleep_staging_model.pth')

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for aug1, aug2 in dataloader:
        aug1 = aug1.unsqueeze(1).to(device)
        output, _ = model(aug1)
        _, predicted = torch.max(output.data, 1)
        total += predicted.size(0)
        correct += (predicted == fixed_labels[:predicted.size(0)]).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')