import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class FreRA(Module):
    def __init__(self, len_sw, device=None, dtype=None):
        super(FreRA, self).__init__()
        print('Initializing FreRA')

        factory_kwargs = {'device': device, 'dtype': dtype}

        # rFFT 后的频率分量数
        self.n_fourier_comp = len_sw // 2 + 1

        # 每个频率一个可学习门控参数
        self.weight = Parameter(torch.empty(self.n_fourier_comp, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.1)

    def get_sampling(self, weight, temperature=0.1, bias=1e-4):
        if self.training:
            eps = torch.rand_like(weight)
            eps = eps.clamp(min=bias, max=1 - bias)
            gate = (torch.log(eps) - torch.log(1 - eps) + weight) / temperature
            return torch.sigmoid(gate)
        else:
            return torch.sigmoid(weight)

    def forward(self, x, temperature=0.1):
        """
        x: (B, C=1, T)
        return: (B, 1, T)
        """
        assert x.dim() == 3, f"Expected (B,1,T), got {x.shape}"

        B, C, T = x.shape
        assert C == 1, "FreRA currently supports single-channel input"

        # → (B, T, 1)
        x = x.transpose(1, 2)

        # FFT: (B, T, 1) → (B, F, 1)
        x_ft = torch.fft.rfft(x, dim=1)

        # 频域门控
        para = self.get_sampling(self.weight, temperature=temperature)  # (F,)
        para = para.view(1, -1, 1)  # (1, F, 1)

        x_ft = x_ft * para

        # IFFT: (B, F, 1) → (B, T, 1)
        x_aug = torch.fft.irfft(x_ft, n=T, dim=1)

        # → (B, 1, T)
        x_aug = x_aug.transpose(1, 2)

        return x_aug


