import torch
from torch import nn

class ChannelAttention1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)   # (B, C, L) -> (B, C, 1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)   # (B, C, L) -> (B, C, 1)
        
        self.fc = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)  # (B, C, 1)


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        # 注意：kernel_size 应为奇数，且 <= L（序列长度）
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, L)
        avg_out = torch.mean(x, dim=1, keepdim=True)    # (B, 1, L)
        concat = torch.cat([max_out, avg_out], dim=1)   # (B, 2, L)
        out = self.conv(concat)                         # (B, 1, L)
        return self.sigmoid(out)                        # (B, 1, L)


class CBAM1D(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3):
        super().__init__()
        self.channel_attn = ChannelAttention1D(channel, reduction)
        self.spatial_attn = SpatialAttention1D(kernel_size)

    def forward(self, x):
        """
        Input:  x of shape (B, C, L)
        Output: same shape (B, C, L)
        """
        residual = x
        x = x * self.channel_attn(x)      # (B, C, L) * (B, C, 1) → (B, C, L)
        x = x * self.spatial_attn(x)      # (B, C, L) * (B, 1, L) → (B, C, L)
        return x + residual