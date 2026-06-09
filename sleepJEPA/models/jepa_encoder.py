import torch
import torch.nn as nn


class EEG1DTransformerEncoder(nn.Module):
    def __init__(
        self,
        patch_len: int = 60,
        seq_len: int = 50,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.patch_proj = nn.Linear(patch_len, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.patch_len)
        x = self.patch_proj(x)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        return x


class Predictor(nn.Module):
    def __init__(self, embed_dim: int = 256, ctx_len: int = 20, tgt_len: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.out_proj = nn.Linear(ctx_len, tgt_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.transpose(1, 2)
        x = self.out_proj(x)
        x = x.transpose(1, 2)
        return x


class JEPAModel(nn.Module):
    """Model components for JEPA framework: context encoder, target encoder, predictor."""
    def __init__(
        self,
        patch_len: int = 60,
        seq_len: int = 50,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        ctx_len: int = 20,
        tgt_len: int = 3,
    ):
        super().__init__()
        self.ctx_encoder = EEG1DTransformerEncoder(
            patch_len=patch_len,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.tgt_encoder = EEG1DTransformerEncoder(
            patch_len=patch_len,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.tgt_encoder.load_state_dict(self.ctx_encoder.state_dict())
        for param in self.tgt_encoder.parameters():
            param.requires_grad = False

        self.predictor = Predictor(embed_dim=embed_dim, ctx_len=ctx_len, tgt_len=tgt_len)
        self.ctx_len = ctx_len
        self.tgt_len = tgt_len

    def forward(self, x: torch.Tensor, ctx_idx: torch.Tensor = None, tgt_idx: torch.Tensor = None):
        x = x.float()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        full_ctx_feat = self.ctx_encoder(x)
        if ctx_idx is None or tgt_idx is None:
            ctx_idx, tgt_idx = self.generate_mask_indices(
                total_len=full_ctx_feat.size(1),
                ctx_num=self.ctx_len,
                tgt_num=self.tgt_len,
            )
            if x.is_cuda:
                ctx_idx = ctx_idx.to(x.device)
                tgt_idx = tgt_idx.to(x.device)
        ctx_feat = full_ctx_feat[:, ctx_idx, :]
        pred_feat = self.predictor(ctx_feat)
        with torch.no_grad():
            full_tgt_feat = self.tgt_encoder(x)
            tgt_feat = full_tgt_feat[:, tgt_idx, :]
        return pred_feat, tgt_feat

    @staticmethod
    def generate_mask_indices(total_len: int = 50, ctx_num: int = 20, tgt_num: int = 3):
        ctx_idx = torch.randperm(total_len)[:ctx_num]
        tgt_idx = torch.randperm(total_len)[:tgt_num]
        return ctx_idx, tgt_idx

    @staticmethod
    def patch_length():
        return 60

    @staticmethod
    def seq_length():
        return 50
