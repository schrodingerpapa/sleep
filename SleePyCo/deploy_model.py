import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import numpy as np
from models.main_model import MainModel


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
                # 直接对整个序列进行平均池化
                avg_pool = F.adaptive_avg_pool1d(x, 1)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                # 使用固定的操作替代动态计算
                # 直接对整个序列进行最大池化
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
    修复模型中的ChannelGate模块以支持ONNX导出
    """

    def replace_channel_gate(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Module):
                if child.__class__.__name__ == 'ChannelGate':
                    # 获取原始参数
                    gate_channels = child.gate_channels
                    # 创建修复版本
                    fixed_gate = ChannelGateFixed(gate_channels)
                    # 复制MLP权重（如果可能）
                    try:
                        fixed_gate.mlp.load_state_dict(child.mlp.state_dict())
                    except:
                        print(f"Warning: Could not copy weights for ChannelGate {name}")
                    # 替换模块
                    setattr(module, name, fixed_gate)
                else:
                    replace_channel_gate(child)

    replace_channel_gate(model)
    return model


def convert_to_onnx(config_path, fused_model_path, onnx_path, input_shape=(10, 1, 3000)):
    """
    将融合后的PyTorch模型转换为ONNX格式

    Args:
        config_path: 配置文件路径
        fused_model_path: 融合模型路径 (.pth文件)
        onnx_path: 输出ONNX模型路径
        input_shape: 输入数据形状 (batch_size, channels, sequence_length)
    """

    # 确保输出目录存在
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # 加载配置文件
    with open(config_path) as config_file:
        config = json.load(config_file)

    # 创建模型实例
    print("Creating model instance...")
    model = MainModel(config)

    # 加载融合后的权重
    print("Loading fused model weights...")
    state_dict = torch.load(fused_model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    # 修复模型以支持ONNX导出
    print("Fixing model for ONNX export...")
    model = fix_model_for_onnx(model)

    model.eval()

    # 创建示例输入
    dummy_input = torch.randn(input_shape)

    # 导出为ONNX格式
    print("Exporting model to ONNX format...")
    try:
        # 使用opset_version=14来支持scaled_dot_product_attention操作
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,  # 升级到14以支持scaled_dot_product_attention
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model successfully exported to ONNX format: {onnx_path}")

    except Exception as e:
        print(f"Error during ONNX export with dynamic axes: {e}")
        print("Trying export with fixed input shape...")

        # 尝试使用固定输入形状导出
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,  # 升级到14
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=None  # 不使用动态轴
            )
            print(f"Model successfully exported to ONNX format: {onnx_path}")
        except Exception as e2:
            print(f"Error during ONNX export with fixed axes: {e2}")
            return False

    # 验证ONNX模型
    print("\nValidating ONNX model...")
    try:
        import onnx
        import onnxruntime as ort

        # 检查ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")

        # 测试推理
        print("Testing inference with ONNX Runtime...")
        ort_session = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(*input_shape).astype(np.float32)
        outputs = ort_session.run(None, {'input': test_input})
        print(f"Inference test successful! Output shape: {outputs[0].shape}")

        return True

    except ImportError:
        print("ONNX or ONNX Runtime not installed. Skipping validation.")
        return True
    except Exception as e:
        print(f"Error during validation: {e}")
        return False


def test_model_inference(config_path, fused_model_path, input_shape=(10, 1, 3000)):
    """
    测试PyTorch模型推理是否正常工作
    """
    print("Testing PyTorch model inference...")

    # 加载配置文件
    with open(config_path) as config_file:
        config = json.load(config_file)

    # 创建模型实例
    model = MainModel(config)

    # 加载融合后的权重
    state_dict = torch.load(fused_model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # 测试推理
    dummy_input = torch.randn(input_shape)
    with torch.no_grad():
        output = model(dummy_input)
        if isinstance(output, list):
            output = output[0]

    print(f"PyTorch model inference successful! Output shape: {output.shape}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert fused PyTorch model to ONNX")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--fused_model", type=str, required=True,
                        help="Path to fused model (.pth file)")
    parser.add_argument("--onnx_model", type=str, required=True,
                        help="Output path for ONNX model")
    parser.add_argument("--input_shape", type=int, nargs=3, default=[10, 1, 3000],
                        help="Input shape (batch_size, channels, sequence_length), default: 10 1 3000")
    parser.add_argument("--test_only", action="store_true",
                        help="Only test model inference without exporting to ONNX")

    args = parser.parse_args()

    # 首先测试PyTorch模型推理
    test_success = test_model_inference(
        config_path=args.config,
        fused_model_path=args.fused_model,
        input_shape=tuple(args.input_shape)
    )

    if args.test_only:
        return

    if test_success:
        success = convert_to_onnx(
            config_path=args.config,
            fused_model_path=args.fused_model,
            onnx_path=args.onnx_model,
            input_shape=tuple(args.input_shape)
        )

        if success:
            print("\nConversion completed successfully!")
            print(f"- Input PyTorch model: {args.fused_model}")
            print(f"- Output ONNX model: {args.onnx_model}")
        else:
            print("\nConversion failed!")
    else:
        print("PyTorch model inference test failed. Aborting ONNX export.")


if __name__ == "__main__":
    main()
