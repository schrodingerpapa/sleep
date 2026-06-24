import torch
import torch.nn as nn
import torch.nn.functional as F

# 在 v2的基础上修改，ECA卷积核大小3——>5
# 卷积层的数目1，2，2，3，2 ——> 1，2，2，2，3

# ==========================================================
# Depthwise Separable Conv Block
# ==========================================================
class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# ==========================================================
# ECA Channel Attention 
# ==========================================================
class ECAGate(nn.Module):
    def __init__(self, channels, k_size=5):
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
        return x * y


# ==========================================================
# 自适应MaxPool
# ==========================================================
class MaxPool1dAdaptive(nn.Module):
    def __init__(self, maxpool_size):
        super().__init__()
        self.maxpool_size = maxpool_size
        self.pool = nn.MaxPool1d(maxpool_size, stride=maxpool_size)

    def forward(self, x):
        _, _, n_samples = x.size()
        if n_samples % self.maxpool_size != 0:
            pad_size = self.maxpool_size - (n_samples % self.maxpool_size)
            left = pad_size // 2
            right = pad_size - left
            x = F.pad(x, (left, right))
        return self.pool(x)


# ==========================================================
#  Lightweight Backbone
# ==========================================================
class SleePyCoUltraLightBackbone(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.training_mode = config['training_params']['mode']
        

        # 轻量通道配置
        self.init_layer = self.make_stage(1, 24, 1, first=True)
        self.layer1 = self.make_stage(24, 48, 2, pool=5)
        self.layer2 = self.make_stage(48, 72, 2, pool=5)
        self.layer3 = self.make_stage(72, 96, 2, pool=5)
        self.layer4 = self.make_stage(96, 128, 3, pool=5)

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
    def make_stage(self, in_ch, out_ch, n_layers, pool=None, first=False):
        layers = []

        if not first and pool is not None:
            layers.append(MaxPool1dAdaptive(pool))

        for i in range(n_layers):
            layers.append(DWConvBlock(in_ch, out_ch))
            in_ch = out_ch

        layers.append(ECAGate(out_ch))  # 轻量注意力

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

        c1 = self.init_layer(x)   # B,24,3000
        c2 = self.layer1(c1)      # B,48,600
        c3 = self.layer2(c2)      # B,72,120
        c4 = self.layer3(c3)      # B,96,24
        c5 = self.layer4(c4)      # B,128,5

        if self.training_mode in ['pretrain', 'FreRA', 'mix_FreRA']:
            out.append(c5)

        elif self.training_mode in ['scratch', 'fullfinetune', 'freezefinetune']:

            p5 = self.conv_c5(c5)
            out.append(p5)

            if self.num_scales > 1:
                p4 = self.conv_c4(c4)
                out.append(p4)

            if self.num_scales > 2:
                p3 = self.conv_c3(c3)
                out.append(p3)

        return out


# ==========================================================
# 5️⃣ 测试
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

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params / 1e6:.2f} M")