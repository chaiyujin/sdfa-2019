import math
import torch
import saber
import numpy as np


class PcaUnprojection(torch.nn.Module):
    def __init__(self, pca_compT, pca_means, trainable):
        super().__init__()
        pca_compT = pca_compT.astype(np.float32)
        pca_means = pca_means.astype(np.float32)
        if trainable:
            self.register_parameter("compT", torch.FloatTensor(pca_compT))
            self.register_parameter("means", torch.FloatTensor(pca_means))
        else:
            self.register_buffer("compT", torch.FloatTensor(pca_compT))
            self.register_buffer("means", torch.FloatTensor(pca_means))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.compT, self.means)


class MultiplicativeNoise(torch.nn.Module):
    def __init__(self, base=1.4, mean=0.0, std=1.0):
        super().__init__()
        self.base, self.mean, self.std = base, mean, std

    def random(self, inputs):
        half = inputs.size(0) // 2
        size = [int(inputs.size(i)) if i < 2 else 1
                for i in range(inputs.dim())]
        # per-feature map basis
        _mean = inputs.new_zeros(size)
        _stdv = inputs.new_zeros(size)
        _mean.fill_(self.mean)
        _stdv.fill_(self.std)
        _norm = torch.normal(_mean, _stdv)
        if inputs.size(0) > 1:
            _norm[half:] = _norm[:half]  # same for adj frames
        return inputs * (self.base ** _norm)

    def forward(self, inputs):
        return self.random(inputs) if self.training else inputs
