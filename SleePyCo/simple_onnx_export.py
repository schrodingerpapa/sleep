# robust_onnx_export.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from models.main_model import MainModel
import argparse
import os


class MaxPool1dFixed(nn.Module):
    """
  修复版本的MaxPool1d，解决ONNX导出问题
  """

    def __init__(self, maxpool_size):
        super(MaxPool1dFixed, self).__init__()
        self.maxpool_size = maxpool_size
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)

    def forward(self, x):
        _, _, n_samples = x.size()
        # 修复动态尺寸问题
        if self.maxpool_size is not None and self.maxpool_size > 0:
            if n_samples % self.maxpool_size != 0:
                pad_size = self.maxpool_size - (n_samples % self.maxpool_size)
                if pad_size > 0:
                    if pad_size % 2 != 0:
                        left_pad = pad_size // 2
                        right_pad = pad_size // 2 + 1
                    else:
                        left_pad = pad_size // 2
                        right_pad = pad_size // 2
                    x = F.pad(x, (left_pad, right_pad), mode='constant')

            x = self.maxpool(x)
        return x


class ChannelGateFixed(nn.Module):
    """
  修复版本的ChannelGate，解决ONNX导出问题
  """

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGateFixed, self).__init__()
        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.pool_types = pool_types

        # 预先定义MLP层
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                # 使用固定的操作替代动态计算
                avg_pool = F.adaptive_avg_pool1d(x, 1)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                # 使用固定的操作替代动态计算
                max_pool = F.adaptive_max_pool1d(x, 1)
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # 使用sigmoid激活
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
        return x * scale


def fix_model_for_onnx(model):
    """
  修复模型以支持ONNX导出
  """

    def replace_modules(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Module):
                # 替换MaxPool1d模块
                if child.__class__.__name__ == 'MaxPool1d':
                    maxpool_size = getattr(child, 'maxpool_size', None)
                    if maxpool_size is not None:
                        fixed_pool = MaxPool1dFixed(maxpool_size)
                        setattr(module, name, fixed_pool)
                # 替换ChannelGate模块
                elif child.__class__.__name__ == 'ChannelGate':
                    gate_channels = child.gate_channels
                    fixed_gate = ChannelGateFixed(gate_channels)
                    try:
                        fixed_gate.mlp.load_state_dict(child.mlp.state_dict())
                    except:
                        print(f"Warning: Could not copy weights for ChannelGate {name}")
                    setattr(module, name, fixed_gate)
                else:
                    replace_modules(child)

    replace_modules(model)
    return model


def export_to_onnx(config_path, model_path, onnx_path, input_shape=(10, 1, 3000)):
    # 加载配置
    with open(config_path) as f:
        config = json.load(f)

    # 创建并加载模型
    print("Creating model...")
    model = MainModel(config)

    print("Loading model weights...")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    # 修复模型以支持ONNX导出
    print("Fixing model for ONNX export...")
    model = fix_model_for_onnx(model)

    model.eval()

    # 创建测试输入
    dummy_input = torch.randn(input_shape)

    # 测试PyTorch推理
    print("Testing PyTorch inference...")
    with torch.no_grad():
        output = model(dummy_input)
        if isinstance(output, list):
            output = output[0]
    print(f"PyTorch output shape: {output.shape}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # 导出ONNX
    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Successfully exported to {onnx_path}")
        return True
    except Exception as e:
        print(f"Export failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    success = export_to_onnx(args.config, args.model, args.output)
    if success:
        print("ONNX export completed!")
    else:
        print("ONNX export failed!")
