import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class RandomGaussianNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        noise = np.random.randn(*x.shape) * self.std
        return x + noise


class RandomTimeShift:
    def __init__(self, max_shift=100):
        self.max_shift = max_shift

    def __call__(self, x):
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(x, shift, axis=-1)


class RandomZeroMasking:
    def __init__(self, mask_ratio=0.15):
        self.mask_ratio = mask_ratio

    def __call__(self, x):
        seq_len = x.shape[-1]
        mask_len = int(seq_len * self.mask_ratio)
        if mask_len <= 0:
            return x
        start = np.random.randint(0, seq_len - mask_len + 1)
        x[..., start : start + mask_len] = 0.0
        return x


class TwoTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x.copy()), self.transform(x.copy())
