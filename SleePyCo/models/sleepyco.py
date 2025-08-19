import torch
import torch.nn as nn
import torch.nn.functional as F


class SleePyCoBackbone(nn.Module): # 模型主干网络
    
    def __init__(self, config):
        super(SleePyCoBackbone, self).__init__()

        self.training_mode = config['training_params']['mode']

        # architecture
        self.init_layer = self.make_layers(in_channels=1, out_channels=64, n_layers=2, maxpool_size=None, first=True)
        self.layer1 = self.make_layers(in_channels=64, out_channels=128, n_layers=2, maxpool_size=5)
        self.layer2 = self.make_layers(in_channels=128, out_channels=192, n_layers=3, maxpool_size=5)
        self.layer3 = self.make_layers(in_channels=192, out_channels=256, n_layers=3, maxpool_size=5)
        self.layer4 = self.make_layers(in_channels=256, out_channels=256, n_layers=3, maxpool_size=5)

        if self.training_mode == 'freezefinetune' or self.training_mode == 'scratch': #训练模式，根据不同模式决定是否使用特征金字塔
            self.fp_dim = config['feature_pyramid']['dim'] # 确定特征金字塔的维度
            self.num_scales = config['feature_pyramid']['num_scales'] # 确定特征金字塔的尺度数量
            self.conv_c5 = nn.Conv1d(256, self.fp_dim, 1, 1, 0)
            # 使用 1x1 卷积层 conv_c5, conv_c4, conv_c3 对不同尺度的特征图进行降维，使其具有相同的维度 fp_dim
            # in_channels=256（Conv5原始通道）
            # out_channels=128（目标通道d_f）
            # kernel_size=1（1×1卷积）

            if self.num_scales > 1:
                self.conv_c4 = nn.Conv1d(256, self.fp_dim, 1, 1, 0)
            
            if self.num_scales > 2:
                self.conv_c3 = nn.Conv1d(192, self.fp_dim, 1, 1, 0)
            
        if config['backbone']['init_weights']:
            self._initialize_weights()

    def _initialize_weights(self): # 初始化权重参数
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # kaiming初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def make_layers(self, in_channels, out_channels, n_layers, maxpool_size, first=False):
        # 构建卷积层
        layers = []
        layers = layers + [MaxPool1d(maxpool_size)] if not first else layers

        for i in range(n_layers):
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            layers += [conv1d, nn.BatchNorm1d(out_channels)]
            if i == n_layers - 1:
                layers += [ChannelGate(in_channels)]
            layers += [nn.PReLU()]
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x): # x: [batch_size, channel, length],前向传播
        out = []
        # 输入数据得到不同尺度的特征图
        c1 = self.init_layer(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        if self.training_mode == 'pretrain':
            out.append(c5) # 预训练只返回最后一层的特征图
        elif self.training_mode in ['scratch', 'fullyfinetune', 'freezefinetune']:
            # 根据训练模式返回不同尺度的特征图
            p5 = self.conv_c5(c5)
            out.append(p5)
            if self.num_scales > 1:
                p4 = self.conv_c4(c4)
                out.append(p4)
            if self.num_scales > 2:
                p3 = self.conv_c3(c3)
                out.append(p3)
        
        return out


class MaxPool1d(nn.Module):
    def __init__(self, maxpool_size):
        super(MaxPool1d, self).__init__()
        self.maxpool_size = maxpool_size
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)

    def forward(self, x):
        _, _, n_samples = x.size()
        if n_samples % self.maxpool_size != 0:
            pad_size = self.maxpool_size - (n_samples % self.maxpool_size)
            if pad_size % 2 != 0:
                left_pad = pad_size // 2
                right_pad = pad_size // 2 + 1
            else:
                left_pad = pad_size // 2
                right_pad = pad_size // 2
            x = F.pad(x, (left_pad, right_pad), mode='constant')

        x = self.maxpool(x)

        return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type=='max':
                max_pool = F.max_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs
