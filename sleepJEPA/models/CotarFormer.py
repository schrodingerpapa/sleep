import torch
import torch.nn as nn
import torch.nn.functional as F


class CotarFormer(nn.Module):
    def __init__(self, config):
        super(CotarFormer, self).__init__()
        self.encoder_depth = config["backbone"]["encoder_depth"]
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(self.encoder_depth)]
        )

    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["backbone"]["ebd_dim"]
        self.norm1 = nn.LayerNorm(self.d_model)
        self.dropout = config["backbone"]["dropout"]
        self.attn = CoTAR(self.d_model, self.d_model // 4)

        self.norm2 = nn.LayerNorm(self.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, int(2 * self.d_model)),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(2 * self.d_model), self.d_model),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        x_att = self.attn(x)
        x = self.norm1(x + x_att)

        x_ln = self.mlp(x)
        x = self.norm2(x + x_ln)

        return x


class CoTAR(nn.Module):
    def __init__(self, d_model, d_core=64):
        super(CoTAR, self).__init__()
        self.d_model = d_model

        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_core)
        self.lin3 = nn.Linear(d_model + d_core, d_model)
        self.lin4 = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape

        core = F.gelu(self.lin1(x))
        core = self.lin2(core)

        weight = F.softmax(core, dim=1)
        core = torch.sum(core * weight, dim=1, keepdim=True).repeat(1, N, 1)

        core_cat = torch.cat([x, core], -1)
        core_cat = F.gelu(self.lin3(core_cat))
        core_cat = self.lin4(core_cat)
        out = core_cat

        return out
