import os
import torch
import json
from models.main_model import MainModel


def average_fold_models(config_path, checkpoint_dir, output_path):
    """
    将10折交叉验证的模型参数进行平均，并保存为一个模型

    Args:
        config_path: 配置文件路径
        checkpoint_dir: 模型检查点目录
        output_path: 输出模型路径
    """
    # 加载配置文件
    with open(config_path) as config_file:
        config = json.load(config_file)

    # 创建模型实例
    model = MainModel(config)
    # 使用DataParallel包装模型，与训练时保持一致
    model = torch.nn.DataParallel(model)

    # 收集所有模型的权重
    all_state_dicts = []
    for fold in range(1, 11):  # 10折交叉验证
        model_path = os.path.join(checkpoint_dir, f'ckpt_fold-{fold:02d}.pth')
        if os.path.exists(model_path):
            # 使用weights_only=True避免安全警告
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            all_state_dicts.append(state_dict)
            print(f"Loaded model from fold {fold}")
        else:
            print(f"Warning: Model for fold {fold} not found at {model_path}")

    if not all_state_dicts:
        raise ValueError("No model checkpoints found!")

    # 计算平均权重
    print("Averaging model weights...")
    averaged_state_dict = {}
    for key in all_state_dicts[0].keys():
        # 收集所有模型在该层的权重
        weights = [state_dict[key] for state_dict in all_state_dicts]

        # 检查数据类型，只对浮点类型进行平均
        if weights[0].dtype in [torch.float16, torch.float32, torch.float64]:
            # 计算平均值
            averaged_state_dict[key] = torch.stack(weights).mean(dim=0)
        else:
            # 对于非浮点类型（如整数、布尔值），直接复制第一个模型的权重
            averaged_state_dict[key] = weights[0]
            print(f"Skipping averaging for {key} with dtype {weights[0].dtype}")

    # 加载平均后的权重到模型
    model.load_state_dict(averaged_state_dict)

    # 保存融合后的模型
    torch.save(averaged_state_dict, output_path)
    print(f"Averaged model saved to {output_path}")

    return model


def average_fold_models_for_inference(config_path, checkpoint_dir, output_path):
    """
    将10折交叉验证的模型参数进行平均，并保存为可用于推理的模型（去除module.前缀）

    Args:
        config_path: 配置文件路径
        checkpoint_dir: 模型检查点目录
        output_path: 输出模型路径
    """
    # 加载配置文件
    with open(config_path) as config_file:
        config = json.load(config_file)

    # 创建模型实例（不使用DataParallel包装）
    model = MainModel(config)

    # 收集所有模型的权重
    all_state_dicts = []
    for fold in range(1, 11):  # 10折交叉验证
        model_path = os.path.join(checkpoint_dir, f'ckpt_fold-{fold:02d}.pth')
        if os.path.exists(model_path):
            # 使用weights_only=True避免安全警告
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            all_state_dicts.append(state_dict)
            print(f"Loaded model from fold {fold}")
        else:
            print(f"Warning: Model for fold {fold} not found at {model_path}")

    if not all_state_dicts:
        raise ValueError("No model checkpoints found!")

    # 计算平均权重
    print("Averaging model weights...")
    averaged_state_dict = {}
    for key in all_state_dicts[0].keys():
        # 收集所有模型在该层的权重
        weights = [state_dict[key] for state_dict in all_state_dicts]

        # 检查数据类型，只对浮点类型进行平均
        if weights[0].dtype in [torch.float16, torch.float32, torch.float64]:
            # 计算平均值
            averaged_state_dict[key] = torch.stack(weights).mean(dim=0)
        else:
            # 对于非浮点类型（如整数、布尔值），直接复制第一个模型的权重
            averaged_state_dict[key] = weights[0]
            # print(f"Skipping averaging for {key} with dtype {weights[0].dtype}")

    # 移除module.前缀以适配推理
    new_state_dict = {}
    for key, value in averaged_state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # 去掉'module.'前缀
        else:
            new_key = key
        new_state_dict[new_key] = value

    # 加载平均后的权重到模型
    model.load_state_dict(new_state_dict)

    # 保存融合后的模型（用于推理）
    torch.save(new_state_dict, output_path)
    print(f"Averaged model for inference saved to {output_path}")

    return model


if __name__ == "__main__":
    # 示例用法
    config_path = "configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json"
    checkpoint_dir = "checkpoints/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune"
    output_path = "checkpoints/fused_model.pth"

    # 创建用于推理的融合模型
    model = average_fold_models_for_inference(config_path, checkpoint_dir, output_path)
