import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class CotarFormer(nn.Module):
    def __init__(self, config):
        super(CotarFormer, self).__init__()
        self.encoder_depth = config['backbone']['encoder_depth']
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) \
                                            for _ in range(self.encoder_depth)])
    def forward(self, x):
        for encoder_layer in self.encoder_layers:
            x= encoder_layer(x)
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['backbone']['ebd_dim']
        self.norm1 = nn.LayerNorm(self.d_model)
        self.dropout = config['backbone']['dropout']
        # self.attn = Attention(configs.d_model, configs.n_heads) # # for ablation
        self.attn = CoTAR(self.d_model, self.d_model//4)

        self.norm2 = nn.LayerNorm(self.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, int(2*self.d_model)),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(2*self.d_model), self.d_model),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        B, N, D = x.shape
        if D != self.d_model:
            x = nn.Linear(D, self.d_model)(x)
        
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
        if D != self.d_model:
            x = nn.Linear(D, self.d_model)(x)

        # MLP
        core = F.gelu(self.lin1(x))
        core = self.lin2(core)

        weight = F.softmax(core, dim=1)
        core = torch.sum(core * weight, dim=1, keepdim=True).repeat(1, N, 1)

        # MLP
        core_cat = torch.cat([x, core], -1)
        core_cat = F.gelu(self.lin3(core_cat))
        core_cat = self.lin4(core_cat)
        out = core_cat

        return out

if __name__ == "__main__":
    # 创建测试输入数据
    import json
    x = torch.randn(50, 32, 3000)  # EEG [batch, channel, length]
    json_path = r"/home/chenlungan/算法模型/SleePyCo/configs/MAE/SleePyCo-Transformer_SL-01_numScales-1_Sleep-EDF-2018_pretrainMAE.json"
    config = json.load(open(json_path, 'r'))
    model = CotarFormer(config)
    # 测试前向传播
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output[0].shape}")  # 输出是一个列表，取第一个元素