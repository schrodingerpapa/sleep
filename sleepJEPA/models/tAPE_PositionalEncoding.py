import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

feature_len_dict = {
    "SleePyCo": [
        [5, 24, 120],
        [10, 48, 240],
        [15, 72, 360],
        [20, 96, 480],
        [24, 120, 600],
        [29, 144, 720],
        [34, 168, 840],
        [39, 192, 960],
        [44, 216, 1080],
        [48, 240, 1200],
    ],
    "XSleepNet": [
        [6, 12, 24],
        [12, 24, 47],
        [18, 36, 71],
        [24, 47, 94],
        [30, 59, 118],
        [36, 71, 141],
        [42, 83, 165],
        [47, 94, 188],
        [53, 106, 211],
        [59, 118, 236],
    ],
    "UTime": [
        [7, 15, 62],
        [15, 31, 125],
        [23, 45, 187],
        [31, 62, 250],
        [39, 78, 312],
        [46, 93, 375],
        [54, 109, 437],
        [62, 125, 500],
        [70, 140, 562],
        [78, 156, 625],
    ],
}


class tAPE(nn.Module):
    def __init__(self, config, emb_size, dropout=0.1):
        super(tAPE, self).__init__()
        self.cfg = config["classifier"]["pos_enc"]
        self.num_scales = config["feature_pyramid"]["num_scales"]

        if self.cfg["dropout"]:
            self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(in_features=emb_size, out_features=emb_size)
        self.act_fn = nn.PReLU()

        if self.num_scales > 1:
            self.max_len = feature_len_dict[config["backbone"]["name"]][
                config["dataset"]["seq_len"] - 1
            ][config["feature_pyramid"]["num_scales"] - 1]
        else:
            self.max_len = 5000

        print("[INFO] Maximum length of pos_enc: {}".format(self.max_len))

        pe = torch.zeros(self.max_len, emb_size)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size)
        )
        scale_factor = emb_size / self.max_len
        pe[:, 0::2] = torch.sin(position * div_term * scale_factor)
        pe[:, 1::2] = torch.cos(position * div_term * scale_factor)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = self.act_fn(self.fc(x))

        if self.num_scales > 1:
            hop = self.max_len // x.size(0)
            pe = self.pe[hop // 2 :: hop, :]
        else:
            pe = self.pe

        if pe.shape[0] != x.size(0):
            pe = pe[: x.size(0), :]

        x = x + pe

        if self.cfg["dropout"]:
            x = self.dropout(x)

        return x


class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size**-0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )

        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out


class Attention_Rel_Scl(nn.Module):
    def __init__(self, config, emb_size, num_heads, dropout):
        super().__init__()
        self.num_scales = config["feature_pyramid"]["num_scales"]
        if self.num_scales > 1:
            self.max_len = feature_len_dict[config["backbone"]["name"]][
                config["dataset"]["seq_len"] - 1
            ][config["feature_pyramid"]["num_scales"] - 1]
        else:
            self.max_len = 5000
        self.num_heads = num_heads
        self.scale = emb_size**-0.5

        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.max_len - 1), num_heads)
        )

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x.transpose(0, 1)
        batch_size, seq_len, _ = x.shape

        k = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )

        attn = torch.matmul(q, k) * self.scale
        position_index = torch.arange(seq_len, device=x.device)
        relative_position = position_index.unsqueeze(1) - position_index.unsqueeze(0)
        relative_position_index = relative_position + (self.max_len - 1)
        relative_position_index = torch.clamp(
            relative_position_index, 0, 2 * self.max_len - 2
        )
        relative_bias = self.relative_bias_table[relative_position_index]
        relative_bias = relative_bias.permute(2, 0, 1).unsqueeze(0)

        attn = attn + relative_bias
        attn = nn.functional.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, -1)
        out = out.transpose(0, 1)
        out = self.to_out(out)
        return out


class eRAPE_TransformerLayer(nn.Module):
    def __init__(self, config, nheads):
        super().__init__()
        self.cfg = config["classifier"]
        self.model_dim = self.cfg["model_dim"]
        self.feedforward_dim = self.cfg["feedforward_dim"]
        self.LayerNorm1 = nn.LayerNorm(self.model_dim, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(self.model_dim, eps=1e-5)
        self.attention_layer = Attention_Rel_Scl(
            config, self.model_dim, nheads, dropout=self.cfg["dropout"]
        )

        self.FeedForward = nn.Sequential(
            nn.Linear(self.model_dim, self.feedforward_dim),
            nn.GELU(),
            nn.Dropout(self.cfg["dropout"]),
            nn.Linear(self.feedforward_dim, self.model_dim),
            nn.Dropout(self.cfg["dropout"]),
        )

    def forward(self, x):
        att = x + self.attention_layer(x)
        att = self.LayerNorm1(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        return out


class eRPE_Transformer(nn.Module):
    def __init__(self, config, nheads, num_encoder_layers, pool="mean"):
        super().__init__()
        self.config = config
        self.model_dim = self.config["classifier"]["model_dim"]
        self.Fix_pos_encode = True
        self.embed_layer = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim, eps=1e-5),
        )
        self.Fix_Position = tAPE(
            config=self.config,
            emb_size=self.model_dim,
            dropout=self.config["classifier"]["pos_enc"]["dropout"],
        )
        self.transformer = nn.Sequential(
            *[
                eRAPE_TransformerLayer(self.config, nheads)
                for _ in range(num_encoder_layers)
            ]
        )
        self.dropout = nn.Dropout(self.config["classifier"]["dropout"])
        self.pool = pool
        if pool == "attn":
            self.w_ha = nn.Linear(self.model_dim, self.model_dim, bias=True)
            self.w_at = nn.Linear(self.model_dim, 1, bias=False)
        self.fc = nn.Linear(self.model_dim, self.config["classifier"]["num_classes"])

    def forward(self, x):
        x = x.transpose(0, 1)
        x_src = self.embed_layer(x)
        if self.Fix_pos_encode != "None":
            x_src = self.Fix_Position(x_src)
        out = self.transformer(x_src)
        out = out.transpose(0, 1)

        if self.pool == "mean":
            out = out.mean(dim=1)
        elif self.pool == "last":
            out = out[:, -1]
        elif self.pool == "attn":
            a_states = torch.tanh(self.w_ha(out))
            alpha = torch.softmax(self.w_at(a_states), dim=1).view(
                out.size(0), 1, out.size(1)
            )
            out = torch.bmm(alpha, a_states).view(out.size(0), -1)
        elif self.pool is None:
            out = out
        else:
            raise NotImplementedError

        if self.config["classifier"]["dropout"]:
            out = self.dropout(out)
        out = self.fc(out)
        return out
