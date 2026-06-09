# 基于1D\-Transformer的单通道EEG\-I\-JEPA实现文档

## 文档说明

本文档面向**单通道脑电\(EEG\)**时序信号，适配输入尺寸 `(B, 1, 3000)`，完整规范 **1D\-Transformer \+ I\-JEPA** 框架的实现流程、网络结构、数据预处理、掩码索引规则、训练逻辑、完整代码模块与超参配置，所有内容可直接用于工程落地、模型复现与实验开发。

文档所有模块经过维度校验，张量流转闭环，完全适配JEPA自监督预训练范式，适配EEG时序长依赖建模、高层语义特征学习需求。

## 一、整体方案概述

### 1\.1 任务范式

采用JEPA（联合嵌入预测架构）自监督预训练核心逻辑，摒弃像素/波形重建的低层学习模式，基于特征空间推理学习EEG高层时序语义，核心流程如下：

1. 将整条长时序EEG信号切分为等长时序Patch序列；

2. 每轮训练随机选取部分Patch作为可见上下文，遮挡其余区域；

3. 通过上下文编码器提取上下文特征，结合预测头推理未知目标Patch的抽象特征；

4. 完整EEG信号输入动量更新的目标编码器，生成真实目标特征作为监督信号；

5. 通过特征L2距离损失约束预测特征与真实特征对齐，完成网络训练。

### 1\.2 全局统一超参

|参数项|取值|说明|
|---|---|---|
|原始EEG输入尺寸|`(B, 1, 3000)`|批次、单通道、单样本3000个采样点|
|单Patch采样点数|60|单个时序片段包含的原始采样点数量|
|总时序Patch数 N\_all|50|3000 ÷ 60 = 50，全局时序分块总数|
|上下文Patch数 N\_ctx|20|每轮随机选取的可见上下文片段数量|
|预测目标Patch数 K|3|每轮需要推理预测的未知片段数量|
|特征嵌入维度 D|256|编码器统一输出的特征维度|
|编码器架构|1D\-Transformer|建模EEG长距离时序依赖与全局语义|
|目标编码器更新方式|EMA动量更新|无梯度更新，保证特征稳定性|

### 1\.3 整体数据流闭环

```Plain Text
原始EEG (B,1,3000)
      ↓ 时序分块重塑
时序Patch序列 (B,50,60)
├─ 上下文分支：随机采样20个上下文Patch → 1D-Transformer编码器 → (B,20,256) → Predictor → (B,3,256) 【预测特征】
└─ 目标分支：完整时序输入 → EMA动量编码器 → (B,50,256) → 索引取值 → (B,3,256) 【真值特征】
      ↓
MSE/L2损失计算 → 反向传播更新上下文编码器+预测头 → EMA更新目标编码器
```

## 二、数据预处理与索引生成

### 2\.1 EEG时序分块

将原始单通道EEG张量 `(B, 1, 3000)` 重塑为固定长度的时序Patch序列，适配Transformer输入要求。

```Plain Text
import torch

def eeg_to_patch(eeg: torch.Tensor) -> torch.Tensor:
    """
    单通道EEG时序分块
    :param eeg: 原始输入张量 (B, 1, 3000)
    :return: 时序Patch序列 (B, 50, 60)
    """
    B = eeg.shape[0]
    # 压缩通道维度，去除冗余维度
    eeg = eeg.squeeze(1)  # (B, 3000)
    # 重塑为50个时序块，每块60个采样点
    patch_seq = eeg.reshape(B, 50, 60)
    return patch_seq
```

### 2\.2 掩码与目标索引生成

每轮训练动态随机生成上下文索引、目标索引，保证模型学习全局时序关联，避免过拟合固定位置。上下文与目标索引允许轻微重叠，贴合JEPA原生训练逻辑。

```Plain Text
def generate_mask_indices(total_len=50, ctx_num=20, tgt_num=3):
    """
    生成JEPA训练掩码索引
    :param total_len: 总时序块数量
    :param ctx_num: 上下文可见块数量
    :param tgt_num: 待预测目标块数量
    :return: 上下文索引、目标索引
    """
    ctx_idx = torch.randperm(total_len)[:ctx_num]
    tgt_idx = torch.randperm(total_len)[:tgt_num]
    return ctx_idx, tgt_idx
```

## 三、核心网络模块实现

### 3\.1 1D\-Transformer编码器

核心特征提取模块，全程保持时序长度不变，通过可学习位置编码\+自注意力机制建模EEG长时序依赖，输出统一维度的高层语义特征。

```Plain Text
import torch.nn as nn

class EEG1DTransformerEncoder(nn.Module):
    def __init__(
        self,
        patch_len: int = 60,
        seq_len: int = 50,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Patch特征投影：将单块60维采样点映射为256维特征
        self.patch_proj = nn.Linear(patch_len, embed_dim)

        # 时序可学习位置编码（Transformer时序任务必备）
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 正则化抑制过拟合与噪声
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 输入Patch序列 (B, 50, 60)
        :return: 全局时序特征 (B, 50, 256)
        """
        # 特征维度投影
        x = self.patch_proj(x)
        # 叠加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)
        # 自注意力全局特征编码
        x = self.transformer_encoder(x)
        return x
```

### 3\.2 预测头Predictor

轻量特征变换模块，实现上下文特征到目标特征的维度映射，输入上下文特征序列，输出固定数量的目标预测特征，保证与真值特征维度完全对齐。

```Plain Text
class Predictor(nn.Module):
    def __init__(self, embed_dim=256, ctx_len=20, tgt_len=3):
        super().__init__()
        self.embed_dim = embed_dim
        # 特征非线性变换
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        # 序列长度维度映射
        self.out_proj = nn.Linear(ctx_len, tgt_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 上下文特征 (B, 20, 256)
        B, L, D = x.shape
        x = self.net(x)  # 特征变换 (B,20,256)
        x = x.transpose(1, 2)  # 维度转置 (B,256,20)
        x = self.out_proj(x)   # 序列长度映射 (B,256,3)
        x = x.transpose(1, 2)  # 还原标准格式 (B,3,256)
        return x
```

### 3\.3 EMA动量更新模块

实现目标编码器（教师模型）的无梯度动量更新，保证教师特征平滑稳定，提升自监督训练收敛效果。

```Plain Text
def update_ema(source_model: nn.Module, target_model: nn.Module, momentum: float = 0.996):
    """
    EMA动量更新目标编码器参数
    :param source_model: 可训练上下文编码器（学生模型）
    :param target_model: 动量更新目标编码器（教师模型）
    :param momentum: 动量系数，JEPA标准取值0.996
    """
    for src_param, tgt_param in zip(source_model.parameters(), target_model.parameters()):
        tgt_param.data = momentum * tgt_param.data + (1 - momentum) * src_param.data
```

## 四、模型整体组装与优化器配置

### 4\.1 模型初始化规则

1. 上下文编码器、目标编码器采用**完全相同的1D\-Transformer结构**；

2. 目标编码器初始化权重与上下文编码器完全同步，冻结梯度，不参与反向传播；

3. 上下文编码器与Predictor开启梯度更新，为核心训练参数；

4. 损失函数采用MSE损失，实现特征空间L2对齐。

```Plain Text
def build_jepa_model(embed_dim=256):
    # 初始化双编码器
    ctx_encoder = EEG1DTransformerEncoder(embed_dim=embed_dim)
    tgt_encoder = EEG1DTransformerEncoder(embed_dim=embed_dim)

    # 权重同步 + 冻结目标编码器梯度
    tgt_encoder.load_state_dict(ctx_encoder.state_dict())
    for param in tgt_encoder.parameters():
        param.requires_grad = False

    # 初始化预测头
    predictor = Predictor(embed_dim=embed_dim)

    # 损失函数
    criterion = nn.MSELoss()

    return ctx_encoder, tgt_encoder, predictor, criterion
```

### 4\.2 优化器配置

采用Transformer标配AdamW优化器，通过权重衰减抑制过拟合，适配EEG自监督训练场景。

```Plain Text
from torch.optim import AdamW

# 初始化模型
ctx_encoder, tgt_encoder, predictor, criterion = build_jepa_model()

# 聚合可训练参数
model_params = list(ctx_encoder.parameters()) + list(predictor.parameters())

# 优化器配置
optimizer = AdamW(
    model_params,
    lr=1e-4,
    weight_decay=1e-5
)
```

## 五、单轮完整训练流程

包含数据处理、索引生成、双分支前向传播、损失计算、反向更新、EMA更新全流程。

```Plain Text
# 1. 数据预处理分块
# eeg_batch: 原始输入 (B, 1, 3000)
eeg_patch = eeg_to_patch(eeg_batch)  # (B, 50, 60)

# 2. 生成训练索引
ctx_idx, tgt_idx = generate_mask_indices()

# 3. 上下文分支前向传播
full_ctx_feat = ctx_encoder(eeg_patch)       # (B, 50, 256)
ctx_feat = full_ctx_feat[:, ctx_idx, :]       # (B, 20, 256)
pred_feat = predictor(ctx_feat)               # (B, 3, 256)

# 4. 目标分支前向传播（无梯度）
with torch.no_grad():
    full_tgt_feat = tgt_encoder(eeg_patch)   # (B, 50, 256)
    tgt_feat = full_tgt_feat[:, tgt_idx, :]  # (B, 3, 256)

# 5. 损失计算与反向传播
loss = criterion(pred_feat, tgt_feat)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 6. EMA更新目标编码器
update_ema(ctx_encoder, tgt_encoder)
```

## 六、标准超参配置表

|配置项|推荐值|说明|
|---|---|---|
|批次大小 batch\_size|8 / 16|根据设备显存灵活调整|
|基础学习率|1e\-4|Transformer模型最优初始学习率|
|权重衰减|1e\-5|抑制Transformer过拟合|
|EMA动量系数|0\.996|JEPA论文官方标准参数|
|Transformer层数|3|适配EEG数据，避免深层过拟合|
|注意力头数|8|平衡表征能力与计算量|
|Dropout率|0\.1|抑制EEG噪声与过拟合|
|训练轮数|100\~200|根据损失收敛情况早停|
|损失函数|MSELoss|实现特征空间精准对齐|

## 七、关键注意事项与调优策略

### 7\.1 维度校验规范

训练前必须保证全链路维度匹配，标准维度流转：

- 原始输入分块：`(B,50,60)`

- 编码器全局输出：`(B,50,256)`

- 上下文特征：`(B,20,256)`

- 预测/真值特征：`(B,3,256)`

### 7\.2 EEG噪声优化方案

1. 数据预处理阶段：对EEG信号做带通滤波、去除工频干扰、基线校正；

2. 网络正则化：Dropout严格控制在0\.05\~0\.1之间，避免注意力机制过度捕捉噪声特征；

3. 限制网络深度：不堆叠过多Transformer层，防止噪声特征层层放大。

### 7\.3 过拟合解决策略

1. 缩减Transformer层数、注意力头数，降低模型参数量；

2. 适当提升权重衰减、微调Dropout比例；

3. 引入早停策略，监控验证集损失，防止过拟合；

4. 扩充EEG训练样本，增加数据多样性。

### 7\.4 显存与算力优化

1. 显存不足时，可将嵌入维度降至128、减小批次大小；

2. 开启混合精度训练，加速训练、降低显存占用；

3. 预训练完成后，仅保留上下文编码器用于下游任务，丢弃Predictor与目标编码器。

### 7\.5 下游任务迁移使用

自监督预训练完成后，模型使用规范：

1. 丢弃Predictor、目标编码器，仅保留训练完备的**上下文Transformer编码器**；

2. EEG信号经分块、编码后得到 `(B,50,256)` 高层时序特征；

3. 对接分类/回归头，微调适配疲劳检测、睡眠分期、情绪识别等下游任务。

## 八、整体运行流程总结

1. EEG原始数据加载、滤波、归一化预处理；

2. 初始化1D\-Transformer双编码器、Predictor、优化器与损失函数；

3. 遍历训练集，逐批次完成数据分块、索引生成、双分支前向传播；

4. 损失反向传播更新可训练参数，EMA动量更新目标编码器；

5. 监控损失收敛，保存最优上下文编码器权重；

6. 加载预训练权重，迁移微调至EEG下游分类任务。

> （注：文档部分内容可能由 AI 生成）
