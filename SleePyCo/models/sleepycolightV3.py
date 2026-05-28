import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Channel Shuffle（0参数）
# ==========================================================
def channel_shuffle(x, groups=2):
    B, C, L = x.size()
    x = x.view(B, groups, C // groups, L)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, L)


# ==========================================================
# 优化版 Depthwise Separable Conv（带 dilation）
# ==========================================================
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_ch, in_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_ch,
            bias=False
        )

        self.pointwise = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()  # 比 PReLU 更稳定

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = channel_shuffle(x)   # ⭐增强通道交互
        x = self.bn(x)
        return self.act(x)


# ==========================================================
# Residual DW Block（关键提升）
# ==========================================================
class ResidualDWBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()

        self.conv = DWConvBlock(in_ch, out_ch, dilation)

        self.shortcut = (
            nn.Identity() if in_ch == out_ch
            else nn.Conv1d(in_ch, out_ch, 1, bias=False)
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# ==========================================================
# 改进 ECA（带残差）
# ==========================================================
class ECAGate(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)              # B,C,1
        y = y.transpose(-1, -2)           # B,1,C
        y = self.conv(y)
        y = y.transpose(-1, -2)           # B,C,1
        y = self.sigmoid(y)

        return x * y + x   # ⭐残差 attention（关键）


# ==========================================================
# Anti-alias Downsampling（替代 MaxPool）
# ==========================================================
class DownsampleBlock(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()

        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=channels,
            bias=False
        )
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ==========================================================
# Lightweight Backbone（优化版）
# ==========================================================
class SleePyCoLightV3Backbone(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.training_mode = config['training_params']['mode']

        # 不同 stage 使用不同 dilation（扩大感受野）
        self.init_layer = self.make_stage(1, 24, 1, first=True, dilation=8)
        self.layer1 = self.make_stage(24, 48, 2, pool=5, dilation=4)
        self.layer2 = self.make_stage(48, 72, 2, pool=5, dilation=2)
        self.layer3 = self.make_stage(72, 96, 3, pool=5, dilation=1)
        self.layer4 = self.make_stage(96, 128, 2, pool=5, dilation=1)

        # Feature Pyramid
        if self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:
            self.fp_dim = config['feature_pyramid']['dim']
            self.num_scales = config['feature_pyramid']['num_scales']

            self.conv_c5 = nn.Conv1d(128, self.fp_dim, 1)

            if self.num_scales > 1:
                self.conv_c4 = nn.Conv1d(96, self.fp_dim, 1)

            if self.num_scales > 2:
                self.conv_c3 = nn.Conv1d(72, self.fp_dim, 1)

        self._initialize_weights()

    # ------------------------------------------------------
    def make_stage(self, in_ch, out_ch, n_layers, pool=None, first=False, dilation=1):
        layers = []

        if not first and pool is not None:
            layers.append(DownsampleBlock(in_ch, pool))  # ⭐替换 MaxPool

        for i in range(n_layers):
            layers.append(ResidualDWBlock(in_ch, out_ch, dilation))
            in_ch = out_ch

        layers.append(ECAGate(out_ch))

        return nn.Sequential(*layers)

    # ------------------------------------------------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------
    def forward(self, x):

        out = []

        c1 = self.init_layer(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        if self.training_mode in ['pretrain', 'FreRA', 'mix_FreRA']:
            out.append(c5)

        elif self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:

            p5 = self.conv_c5(c5)

            if self.num_scales > 1:
                p4 = self.conv_c4(c4)
                p4 = p4 + F.interpolate(p5, size=p4.shape[-1], mode='nearest')  # ⭐FPN融合

            if self.num_scales > 2:
                p3 = self.conv_c3(c3)
                p3 = p3 + F.interpolate(p4, size=p3.shape[-1], mode='nearest')

            out.append(p5)

            if self.num_scales > 1:
                out.append(p4)

            if self.num_scales > 2:
                out.append(p3)

        return out


# ==========================================================
# 测试
# ==========================================================
if __name__ == "__main__":

    config = {
        'training_params': {'mode': 'scratch'},
        'feature_pyramid': {'dim': 128, 'num_scales': 3}
    }

    x = torch.randn(8, 1, 3000)
    model = SleePyCoUltraLightBackbone(config)

    with torch.no_grad():
        outputs = model(x)

    print("Input:", x.shape)
    for i, o in enumerate(outputs):
        print(f"Output {i}:", o.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params / 1e6:.6f} M")