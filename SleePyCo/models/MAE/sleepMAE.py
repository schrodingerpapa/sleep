import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F


# 将项目根目录加入 Python 搜索路径
sys.path.append('/home/chenlungan/算法模型/SleePyCo')

# 使用绝对导入
from models.CotarFormer import CotarFormer
from loss import NTXentLoss
from models.classifiers import get_classifier

class SleepMAE(nn.Module):
    def __init__(self, config, norm_layer=False):
        super().__init__()
        self.config = config
        self.training_mode = config['training_params']['mode']
        self.seq_len = config['dataset']['seq_len']

        self.d_model = self.config["backbone"]["ebd_dim"]
        self.num_segment = self.config["MAE"]["num_segements"]
        self.num_patch = self.config["MAE"]["num_patches"]


        # 预训练patch数量
        self.pretrain_total_patch = self.num_segment * self.num_patch
        self.patch_size = 3000 // self.pretrain_total_patch

        # -------- Patch Embedding --------
        self.ebd_projection = nn.Linear(self.patch_size, self.d_model)
        self.norm_layer = nn.LayerNorm(self.d_model) if norm_layer else nn.Identity()

        # -------- Masking --------
        self.mask_ratio = self.config["MAE"]["mask_ratio"]
        self.num_mask = int(self.pretrain_total_patch * self.mask_ratio)

        # -------- Mask Token --------
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # -------- Position Embedding --------
        self.pos_embed = nn.Parameter(torch.zeros(1, self.pretrain_total_patch, self.d_model))

        # -------- Transformer Encoder --------
        self.encoder = CotarFormer(config)

        # 对比学习投影头
        self.alpha = self.config["training_params"]["alpha"]
        self.projector = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.contrastive_loss = NTXentLoss(temperature=0.5)
        self.rec_loss = nn.MSELoss()  # 重建损失MSE


        if self.training_mode == "pretrain" or self.training_mode == "pretrainMAE":
            self.decoder = nn.Sequential(
                nn.Linear(self.d_model, self.d_model//2,bias = True),
                nn.ReLU(),
                nn.Linear(self.d_model//2, self.patch_size,bias = True))
        elif self.training_mode in ["finetune","freezefinetune","fullfinetune"]:
            self.classifier = get_classifier(config)

        self.initialize_weights()
   
    def interpolate_pos_encoding(self, num_patch):
        if num_patch == self.pos_embed.shape[1]:
            return self.pos_embed

        pos_embed = F.interpolate(
            self.pos_embed.permute(0, 2, 1),
            size=num_patch,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)

        return pos_embed

    def random_masking(self, x, mask_ratio):
        n, l, d = x.shape  # batch, length, dim length表示patch的数量
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        # generate the binary mask 0 is keep, 1 isremove
        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask.bool(), ids_restore



    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
    
    def encode_with_mask(self, x, target):
        # 随机掩码
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio) 

        # 添加mask token  
        B, P_keep, D = x_masked.shape
        mask_tokens = self.mask_token.repeat(B, self.pretrain_total_patch - P_keep, 1)

        x_ = torch.cat([x_masked, mask_tokens], dim=1)
        x_encoder = torch.gather(x_,dim=1,index=ids_restore.unsqueeze(-1).repeat(1, 1, D) ) 

        # 编码器输入
        latent = self.encoder(x_encoder) 
        # 只取mask位置
        mask = mask.bool()
        latent_masked = latent[mask].view(B, -1, D)

        # 掩码patch解码预测
        pred = self.decoder(latent_masked)

        # 取真实mask patch
        target_masked = target[mask].view(B, -1, self.patch_size)

        recon_loss = self.rec_loss(pred, target_masked)
        
        return latent, recon_loss
    

    def forward(self, x):
        # print(x.shape)
        B, _, L = x.shape
        assert L % self.patch_size == 0, "Input length must be divisible by patch_size"
        num_patch = L // self.patch_size
        # 分块
        x = x.view(B, num_patch, self.patch_size)
        target = x.clone()  # 用于计算重建损失

        # patch embedding
        x = self.ebd_projection(x)                         # [B, P, d_model]
        # 动态位置编码
        x = x + self.interpolate_pos_encoding(num_patch)   # [B, P, d_model]
        x = self.norm_layer(x)
 
        if self.training_mode in ["pretrain", "pretrainMAE"]:
            # 双视图
            latent1, recon_loss1 = self.encode_with_mask(x, target)
            latent2, recon_loss2 = self.encode_with_mask(x, target)

            recon_loss = recon_loss1 + recon_loss2
            # 全局平均池化后投影
            z1 = self.projector(latent1.mean(dim=1))  # [B, 128]
            z2 = self.projector(latent2.mean(dim=1))  # [B,128]

            # contrastive_loss,_ = self.contrastive_loss(z1, z2)
            # loss = recon_loss + self.alpha * contrastive_loss

            return recon_loss, z1 , z2  
            
        elif self.training_mode in ["finetune","freezefinetune","fullfinetune"]:
            latent = self.encoder(x)
            latent = latent.view(B,self.seq_len,-1,self.d_model) # [B,seq_len,num_patch, d_model]
            latent = latent.mean(dim=2) # [B,seq_len,d_model]

            out = self.classifier(latent)

            return out
        
        else:
            raise ValueError("Invalid training mode")


        
if __name__ == "__main__":
    # 创建测试输入数据
    import json
    x = torch.randn(50, 1, 3000)  # EEG [batch, channel, length]
    json_path = r"/home/chenlungan/算法模型/SleePyCo/configs/MAE/MAE-AttentionGRU_SL-10_numScales-3_Sleep-EDF-2018_freezefinetune.json"
    config = json.load(open(json_path, 'r'))
    traning_mode = config['training_params']['mode']
    print(f"Training mode: {traning_mode}")
    # 测试前向传播
    with torch.no_grad():
        if traning_mode in ["pretrain", "pretrainMAE"]:
            model = SleepMAE(config)
            loss, recon_loss, contrastive_loss = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Loss: {loss}, Recon loss: {recon_loss}, Contrastive loss: {contrastive_loss}")

        elif traning_mode in ["finetune","freezefinetune"]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SleepMAE(config).to(device)
            x = x.to(device)
            out = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {out.shape}")
            print(f"Output[0]: {out[0]}")
        else:
            raise ValueError("Invalid training mode")

        
