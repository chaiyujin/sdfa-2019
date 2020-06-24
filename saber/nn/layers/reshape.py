import re
import torch
from copy import deepcopy


class Flatten(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x.view(-1, x.size(-1))


class Permute(torch.nn.Module):
    def __init__(self, permutation, **kwargs):
        super().__init__()
        self._permute = permutation

    def forward(self, x):
        return x.permute(*self._permute).contiguous()


class Transpose(torch.nn.Module):
    def __init__(self, dim_a, dim_b, **kwargs):
        super().__init__()
        self._dim_a = dim_a
        self._dim_b = dim_b

    def forward(self, x):
        return x.transpose(self._dim_a, self._dim_b).contiguous()


class Squeeze(torch.nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self._dim = dim

    def forward(self, x):
        assert x.size(self._dim) == 1
        return x.squeeze(self._dim)


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self._dim = dim

    def forward(self, x):
        return x.unsqueeze(self._dim)


class View(torch.nn.Module):
    _dim_re = re.compile(r"^d(\d+)$")

    def __init__(self, shape, **kwargs):
        super().__init__()
        self._shape = []
        for x in shape:
            assert isinstance(x, (int, str))
            if isinstance(x, int):
                self._shape.append(x)
            else:
                x = x.lower()
                assert self._dim_re.match(x) is not None
                self._shape.append(x)

    def forward(self, inputs):
        shape = [
            x if isinstance(x, int) else inputs.shape[int(self._dim_re.match(x).group(1))]
            for x in self._shape
        ]
        return inputs.contiguous().view(*shape)
