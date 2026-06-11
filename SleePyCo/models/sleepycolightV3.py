import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Utility modules
# ==========================================================
class DropPath(nn.Module):
    """Stochastic depth for residual branches."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class ECAGate(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)       # B,C,1
        y = y.transpose(-1, -2)    # B,1,C
        y = self.conv(y)
        y = y.transpose(-1, -2)    # B,C,1
        y = self.sigmoid(y)
        return x * y


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
# Residual large-kernel depthwise block
# ==========================================================
class ResidualLargeKernelDWBlock(nn.Module):
    """A lightweight ConvNeXt-style 1D block for EEG morphology encoding."""

    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=15,
        dilation=1,
        expansion=2,
        eca_kernel=3,
        drop_path=0.0,
        layer_scale_init=1e-3,
    ):
        super().__init__()

        hidden_ch = int(out_ch * expansion)
        padding = dilation * (kernel_size - 1) // 2

        self.dwconv = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(in_ch)

        self.pwconv1 = nn.Conv1d(in_ch, hidden_ch, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.eca = ECAGate(out_ch, k_size=eca_kernel)

        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )

        self.drop_path = DropPath(drop_path)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(out_ch))

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.bn2(x)
        x = self.eca(x)
        x = self.gamma.view(1, -1, 1) * x

        return self.act(residual + self.drop_path(x))


# ==========================================================
# Selective top-down feature pyramid
# ==========================================================
class GatedFPN1D(nn.Module):
    def __init__(self, in_channels, fp_dim, num_scales):
        super().__init__()
        if not 1 <= num_scales <= len(in_channels):
            raise ValueError(f"num_scales must be in [1, {len(in_channels)}], got {num_scales}")

        self.num_scales = num_scales
        self.lateral = nn.ModuleList(
            [nn.Conv1d(ch, fp_dim, kernel_size=1, bias=False) for ch in in_channels[:num_scales]]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(fp_dim, fp_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm1d(fp_dim),
                    nn.GELU(),
                )
                for _ in range(num_scales)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(fp_dim * 2, fp_dim, kernel_size=1, bias=True),
                    nn.Sigmoid(),
                )
                for _ in range(max(0, num_scales - 1))
            ]
        )

    def forward(self, features):
        # Features are ordered from coarse to fine: [c5, c4, c3].
        laterals = [
            lateral(feat)
            for lateral, feat in zip(self.lateral, features[: self.num_scales])
        ]

        outputs = [None] * self.num_scales
        outputs[0] = self.smooth[0](laterals[0])

        prev = outputs[0]
        for i in range(1, self.num_scales):
            up = F.interpolate(prev, size=laterals[i].shape[-1], mode="nearest")
            gate = self.gates[i - 1](torch.cat([laterals[i], up], dim=1))
            fused = laterals[i] + gate * up
            outputs[i] = self.smooth[i](fused)
            prev = outputs[i]

        return outputs


# ==========================================================
# Enhanced lightweight backbone
# ==========================================================
class SleePyCoLightV3Backbone(nn.Module):
    """Drop-in enhanced backbone for lightweight sleep staging experiments."""

    def __init__(self, config):
        super().__init__()

        self.training_mode = config["training_params"]["mode"]
        backbone_cfg = config.get("backbone", {})

        channels = backbone_cfg.get("channels", [24, 48, 72, 96, 128])
        depths = backbone_cfg.get("depths", [1, 2, 2, 3, 2])
        kernel_sizes = backbone_cfg.get("kernel_sizes", [15, 15, 15, 7, 5])
        stage_dilations = backbone_cfg.get(
            "stage_dilations",
            [
                [1],
                [1, 2],
                [1, 2],
                [1, 2, 4],
                [1, 2],
            ],
        )
        expansion = backbone_cfg.get("expansion", 2)
        eca_kernel = backbone_cfg.get("eca_kernel", 3)
        drop_path_rate = backbone_cfg.get("drop_path_rate", 0.05)
        layer_scale_init = backbone_cfg.get("layer_scale_init", 1e-3)

        if len(channels) != 5 or len(depths) != 5 or len(kernel_sizes) != 5:
            raise ValueError("channels, depths, and kernel_sizes must each contain 5 values")

        total_blocks = sum(depths)
        drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        drop_idx = 0

        self.init_layer, drop_idx = self.make_stage(
            1,
            channels[0],
            depths[0],
            kernel_size=kernel_sizes[0],
            dilations=stage_dilations[0],
            expansion=expansion,
            eca_kernel=eca_kernel,
            layer_scale_init=layer_scale_init,
            drop_rates=drop_rates,
            drop_idx=drop_idx,
            first=True,
        )
        self.layer1, drop_idx = self.make_stage(
            channels[0],
            channels[1],
            depths[1],
            pool=5,
            kernel_size=kernel_sizes[1],
            dilations=stage_dilations[1],
            expansion=expansion,
            eca_kernel=eca_kernel,
            layer_scale_init=layer_scale_init,
            drop_rates=drop_rates,
            drop_idx=drop_idx,
        )
        self.layer2, drop_idx = self.make_stage(
            channels[1],
            channels[2],
            depths[2],
            pool=5,
            kernel_size=kernel_sizes[2],
            dilations=stage_dilations[2],
            expansion=expansion,
            eca_kernel=eca_kernel,
            layer_scale_init=layer_scale_init,
            drop_rates=drop_rates,
            drop_idx=drop_idx,
        )
        self.layer3, drop_idx = self.make_stage(
            channels[2],
            channels[3],
            depths[3],
            pool=5,
            kernel_size=kernel_sizes[3],
            dilations=stage_dilations[3],
            expansion=expansion,
            eca_kernel=eca_kernel,
            layer_scale_init=layer_scale_init,
            drop_rates=drop_rates,
            drop_idx=drop_idx,
        )
        self.layer4, _ = self.make_stage(
            channels[3],
            channels[4],
            depths[4],
            pool=5,
            kernel_size=kernel_sizes[4],
            dilations=stage_dilations[4],
            expansion=expansion,
            eca_kernel=eca_kernel,
            layer_scale_init=layer_scale_init,
            drop_rates=drop_rates,
            drop_idx=drop_idx,
        )

        if self.training_mode in ["scratch", "fullfinetune", "freezefinetune"]:
            self.fp_dim = config["feature_pyramid"]["dim"]
            self.num_scales = config["feature_pyramid"]["num_scales"]
            self.fpn = GatedFPN1D(
                in_channels=[channels[4], channels[3], channels[2]],
                fp_dim=self.fp_dim,
                num_scales=self.num_scales,
            )

        self._initialize_weights()

    def make_stage(
        self,
        in_ch,
        out_ch,
        n_layers,
        kernel_size,
        dilations,
        expansion,
        eca_kernel,
        layer_scale_init,
        drop_rates,
        drop_idx,
        pool=None,
        first=False,
    ):
        layers = []

        if not first and pool is not None:
            layers.append(MaxPool1dAdaptive(pool))

        for i in range(n_layers):
            layers.append(
                ResidualLargeKernelDWBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    expansion=expansion,
                    eca_kernel=eca_kernel,
                    drop_path=drop_rates[drop_idx],
                    layer_scale_init=layer_scale_init,
                )
            )
            in_ch = out_ch
            drop_idx += 1

        return nn.Sequential(*layers), drop_idx

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        c1 = self.init_layer(x)   # B,24,3000
        c2 = self.layer1(c1)      # B,48,600
        c3 = self.layer2(c2)      # B,72,120
        c4 = self.layer3(c3)      # B,96,24
        c5 = self.layer4(c4)      # B,128,5

        if self.training_mode in ["pretrain", "FreRA", "mix_FreRA"]:
            return [c5]

        if self.training_mode in ["scratch", "fullfinetune", "freezefinetune"]:
            return self.fpn([c5, c4, c3])

        raise ValueError(f"Unsupported training mode: {self.training_mode}")


if __name__ == "__main__":
    config = {
        "training_params": {"mode": "scratch"},
        "feature_pyramid": {"dim": 128, "num_scales": 3},
        "backbone": {
            "drop_path_rate": 0.05,
            "expansion": 2,
        },
    }

    x = torch.randn(8, 1, 3000)
    model = SleePyCoUltraLightBackboneV3(config)

    with torch.no_grad():
        outputs = model(x)

    print("Input:", x.shape)
    for i, o in enumerate(outputs):
        print(f"Output {i}:", o.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params / 1e6:.3f} M")
