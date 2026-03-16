from torch import nn


class PatchEmbedding(nn.Module):
    """Patchify time series."""

    def __init__(self, config, norm_layer=False):
        super().__init__()
        self.config = config
        self.ebd_dim = self.config["backbone"]["ebd_dim"]
        self.num_patch = self.config["MAE"]["num_patches"]*self.config["MAE"]["num_segments"]  # the total number of patches
        self.patch_size = 3000 // self.num_patch
        
        self.ebd_projection = nn.Linear(self.patch_size, self.ebd_dim)
        
        self.norm_layer = nn.LayerNorm(self.ebd_dim) if norm_layer is True else nn.Identity()

    def forward(self, x):
        out = self.ebd_projection(x)
        out = self.norm_layer(out)
        return out