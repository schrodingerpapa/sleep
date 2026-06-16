# 训练入口说明

新实验建议统一使用根目录下的训练入口：

```bash
python train.py <project> <task> [legacy script args...]
```

原有训练脚本仍然保留在原位置。`train.py` 只负责选择对应脚本、切换到项目目录，并把剩余参数原样转发给旧训练脚本。

## 常用命令

查看所有已注册的训练任务：

```bash
python train.py --list
```

运行 SleePyCo 对比学习预训练：

```bash
python train.py sleepyco crl --config configs/your_config.json --gpu 0
```

运行 SleePyCo MTCL 微调：

```bash
python train.py sleepyco mtcl --config configs/your_config.json --gpu 0
```

运行 SleePyCo 序列到序列微调：

```bash
python train.py sleepyco seq2seq --config configs/SleePyCo-AttentionGRUSeq2Seq_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json --fold 1 --gpu 0
```

运行 SleepJEPA 预训练：

```bash
python train.py jepa pretrain --config configs/your_config.json --fold 1 --gpu 0
```

运行 SleepJEPA 微调：

```bash
python train.py jepa finetune --config configs/your_config.json --fold 1 --gpu 0
```

`--config`、`--pretrain_checkpoint` 这类路径参数既可以写成相对于所选项目目录的路径，也可以写成相对于仓库根目录的路径。

## 已注册任务

| 项目 | 任务名 | 对应旧脚本 | 用途 |
| --- | --- | --- | --- |
| `sleepyco` | `crl` | `SleePyCo/train_crl.py` | 有标签对比表示预训练。 |
| `sleepyco` | `crl-frea` | `SleePyCo/train_crl_FreRA.py` | 使用 FreRA 频域增强的对比预训练。 |
| `sleepyco` | `crl-mix` | `SleePyCo/train_crl_mix.py` | 混合时域增强和 FreRA 频域增强的对比预训练。 |
| `sleepyco` | `no-label` | `SleePyCo/train_no_label.py` | 无标签自监督对比预训练。 |
| `sleepyco` | `mtcl` | `SleePyCo/train_mtcl.py` | 标准 MTCL 微调 / 冻结微调。 |
| `sleepyco` | `seq2seq` | `SleePyCo/train_seq2seq.py` | 对窗口内每个睡眠帧都输出预测的序列到序列微调。 |
| `sleepyco` | `scratch` | `SleePyCo/train_scratch.py` | 从零开始的有监督训练。 |
| `sleepyco` | `mtcl-fullfinetune` | `SleePyCo/train_mtcl_fullfinetune.py` | MTCL 全量微调变体。 |
| `sleepyco` | `mtcl-fullfinetune-2048` | `SleePyCo/train_mtcl_fullfinetune2048.py` | 2048 设置下的 MTCL 全量微调变体。 |
| `sleepyco` | `mae` | `SleePyCo/train_mae.py` | Masked Autoencoder 预训练。 |
| `sleepyco` | `finetune-mae` | `SleePyCo/train_finetuneMAE.py` | 对预训练 MAE 模型进行微调。 |
| `sleepyco` | `fullfinetune-mae` | `SleePyCo/train_fullfinetuneMAE.py` | 对预训练 MAE 模型进行全量微调。 |
| `jepa` | `pretrain` | `sleepJEPA/train_pretrain.py` | JEPA 自监督预训练。 |
| `jepa` | `finetune` | `sleepJEPA/train_finetune.py` | JEPA 下游任务微调。 |

## 命名和维护约定

- 新实验优先使用 `python train.py <project> <task>` 启动。
- 旧训练脚本继续保留，用于兼容已有记录、配置和 checkpoint。
- 新增训练变体时，优先注册到 `train.py` 中，再判断是否需要单独脚本。
- 新配置和新训练脚本中尽量避免硬编码绝对路径。

## 服务器训练性能建议

当前 SleePyCo/Seq2Seq 训练默认模型较小，直接使用 8 张 2080Ti 通过
`DataParallel` 训练时，可能出现显存被占用但 GPU 利用率很低的情况。
这通常是单进程调度、CPU 数据准备或 batch 太小导致的，不一定是 GPU
计算能力不足。

建议按下面顺序排查：

1. 先用单卡或双卡跑通，并把 batch size 调大到显存占用较充分。

```bash
python train.py sleepyco seq2seq --config configs/SleePyCo-AttentionGRUSeq2Seq_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json --fold 1 --gpu 0 --num-workers 8 --amp --benchmark
```

2. 单卡吞吐正常后，再尝试双卡或四卡。

```bash
python train.py sleepyco seq2seq --config configs/SleePyCo-AttentionGRUSeq2Seq_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json --fold 1 --gpu 0,1 --num-workers 8 --amp --benchmark
```

3. 不建议一开始就用 `--gpu 0,1,2,3,4,5,6,7`。当前代码使用
`DataParallel`，8 卡时主进程 scatter/gather 开销明显，小模型反而可能更慢。

4. 观察瓶颈时优先看：

```bash
nvidia-smi dmon -s pucm
top
iostat -xz 1
```

如果 GPU util 低、CPU 单核很高，多半是 `DataParallel` 主线程或数据处理瓶颈。
如果磁盘 util 高，则优先把 `.npz` 数据放到本地 SSD，避免网络盘读取。

## Seq2Seq 分类头

`SleePyCo/train_seq2seq.py` 默认在多个 feature pyramid 尺度之间共享同一个
seq2seq 分类头，配置项为：

```json
"classifier": {
  "share_across_scales": true
}
```

如果需要做消融实验，可以改成 `false`，此时每个尺度会各自创建一个分类头，
分类头参数量也会按 `feature_pyramid.num_scales` 放大。

Seq2Seq 验证和测试默认会把滑动窗口中重复出现的同一个 epoch 进行合并：

```json
"training_params": {
  "eval_strategy": "mean_logits"
}
```

`mean_logits` 会按 `(record, epoch_idx)` 聚合同一个 epoch 的多次预测，再用平均
logits 计算指标，避免滑动窗口评估时重复计数。需要复现旧逻辑时可改为
`window_flatten`。

评估已经训练好的 10 个 fold 时，使用独立脚本，不会重新训练：

```bash
cd SleePyCo
python evaluate_all_folds_seq2seq.py \
  --config configs/SleePyCo-AttentionGRUSeq2Seq_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json \
  --folds 1-10 \
  --gpu 0,1,2,3 \
  --num-workers 16 \
  --amp \
  --benchmark
```

该脚本会先输出每个 fold 的独立测试结果，最后输出所有 fold 拼接后的 pooled
result。默认同样使用 `mean_logits` 合并滑动窗口重复预测。


## 模型参数保存和加载
