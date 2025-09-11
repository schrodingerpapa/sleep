import numpy as np
import matplotlib.pyplot as plt

# 加载损失数据（包含字典的数组）
losses = np.load('losses_fold-03.npy', allow_pickle=True).item()  # .item() 将数组转换为字典

# 提取训练损失和验证损失（直接通过键访问）
train_losses = losses['train_losses']  # 训练损失数组
val_losses = losses['val_losses']      # 验证损失数组
time = losses['training_time'] # 训练时长

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')


# 添加标签和标题
plt.xlabel('Epoch/Step', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.title('Training & Validation Loss Curve (Fold 01)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(linestyle='--', alpha=0.7)
plt.show()

# # 保存图片
plt.savefig('loss_curve_fold-3.png', dpi=300, bbox_inches='tight')
print("损失曲线已保存为 loss_curve_fold-01.png（包含训练和验证损失）")
print("训练时长为：", time)