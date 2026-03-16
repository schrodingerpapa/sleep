import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim=128, d_model=128, num_patches=250):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))

    def forward(self, x):
        # x: [B, L, input_dim]  （假设已经切分为patch）
        x = self.proj(x)                # [B, L, d_model]
        x = x + self.pos_embed           # 添加位置编码
        return x


