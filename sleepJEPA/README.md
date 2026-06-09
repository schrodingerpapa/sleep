# SleepJEPA

基于 JEPA 联合预测框架的单通道 EEG 自监督预训练与微调项目结构。该项目参考 `SleePyCo` 的数据划分、配置加载、训练入口与 checkpoint 管理实现。

## 目录结构

- `configs/`：配置文件
- `checkpoints/`：训练保存的模型权重
- `results/`：训练结果与日志
- `loader.py`：数据集加载与划分逻辑
- `models/jepa.py`：1D Transformer + JEPA 模型实现
- `train_pretrain.py`：自监督预训练入口
- `train_finetune.py`：下游任务微调入口
- `utils.py`：随机数、进度条与保存工具
- `transforms.py`：简单数据增强操作

## 快速使用

### 运行环境

本项目建议在虚拟环境 `sleep_mamba` 中执行：

```bash
conda activate sleep_mamba
python train_pretrain.py --config configs/jepa_pretrain_sleepedf.json --fold 1 --gpu 0
```

### 1. 预训练

```bash
python train_pretrain.py --config configs/jepa_pretrain_sleepedf.json --fold 1 --gpu 0
```

### 2. 微调

```bash
python train_finetune.py --config configs/jepa_finetune_sleepedf.json --pretrain_checkpoint checkpoints/JEPA-Transformer_SL-01_Sleep-EDF-2018_pretrain/best_ckpt_epoch_050.pth --fold 1 --gpu 0
```

### 迭代级早停说明

JEPA 预训练的验证与早停应基于训练迭代次数而不是单纯 epoch：

- `training_params.val_period` 以迭代步数为单位，默认值 `1000`，表示每 1000 个 batch 进行一次验证。
- `training_params.early_stopping.patience` 表示连续多少次验证结果未改善后停止训练，当前默认 `5` 次。

这种设置适合自监督 JEPA 预训练，因为模型需要在多次采样和数据增强后观察验证损失趋势，而不是依赖较长的 epoch 间隔。

## 配置说明

- `dataset.root_dir`：数据集根目录，需要根据本地实际路径修改
- `dataset.name`：数据集名称，当前 loader 支持 `Sleep-EDF-2018`、`Sleep-EDF-2013`，以及通用的文件名划分逻辑
- `model`：Transformer 与 JEPA 模型结构配置
- `training_params`：训练模式、学习率、权重衰减、batch size、是否使用验证集等

## 数据划分

加载方式参考 `SleePyCo/loader.py`：

1. 优先读取 `split_idx/idx_<dataset>.json` 或 `split_idx/idx_<dataset>.npy`
2. 如果未找到 split 文件，则根据文件索引按 fold 做简单训练/验证/测试拆分

## 注意

- 当前实现主要完成框架结构搭建，数据路径与具体 split 文件应根据实际实验环境补充
- 预训练模型权重保存为 `checkpoints/<config.name>/ckpt_epoch_*.pth`
- 微调阶段仅加载 JEPA 上下文编码器权重作为骨干网络
