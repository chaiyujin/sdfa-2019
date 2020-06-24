import torch
from .extend import ILayerExtended
from .linear import FeatureProjection
from .. import functions as fn


class Conv1d(torch.nn.Conv1d, ILayerExtended):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding="same", dilation=1, groups=1, bias=True, **kwargs
    ):
        _pad_val, _pad_mode = fn.check_padding(padding)
        # init torch module
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding=_pad_val, dilation=dilation, groups=groups, bias=bias
        )
        # init extension
        self._ext_init(out_channels, batch_norm_type="1d", **kwargs)
        self._pad_mode = _pad_mode

    def forward(self, x, incremental_state=None):
        x = self._ext_prev_module(x)
        if self._pad_mode is not None:
            x = fn.conv_pad(x, self.kernel_size[0], self.stride[0],
                            self.dilation[0], self._pad_mode)
        x = super().forward(x)
        x = self._ext_post_module(x)
        return x


class ConvTranspose1d(torch.nn.ConvTranspose1d, ILayerExtended):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding="same", output_padding=0, dilation=1, groups=1,
        bias=True, want_size=None, **kwargs
    ):
        _pad_val, _pad_mode = fn.check_padding(padding)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=_pad_val, output_padding=output_padding,
                         groups=groups, bias=bias, dilation=dilation)
        self._ext_init(out_channels, batch_norm_type="1d", **kwargs)

        if isinstance(_pad_mode, str):
            assert _pad_mode in ["same", "valid", "causal"]
            assert want_size is None or isinstance(want_size, int),\
                "'want_size' should be {}.".format(type(int))
            self._want_size = want_size
            self._pad_mode = _pad_mode
        else:
            # using original padding, output_padding
            assert isinstance(padding, int)
            self._want_size = None

    def forward(self, x):
        x = self._ext_prev_module(x)
        x = super().forward(x)
        if self._want_size is not None:
            x = fn.conv_unpad(x, self._want_size, self.kernel_size[0], self.stride[0],
                              self.dilation[0], self._pad_mode)
        x = self._ext_post_module(x)
        return x


class Pool1d(torch.nn.Module):
    def __init__(
        self, mode, kernel_size, stride=None, padding="same", ceil_mode=False,
        dilation=1, return_indices=False,  # max specific
        count_include_pad=True,  # pool specific
    ):
        super().__init__()
        assert mode in ["max", "avg"]
        _pad_val, self._pad_mode = fn.check_padding(padding)
        # build
        self._ksz = kernel_size
        self._hop = stride or kernel_size
        self._dil = dilation
        if mode == "avg":
            self._dil = 1
            self._pool = torch.nn.AvgPool1d(
                kernel_size, stride,
                padding=_pad_val,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad
            )
        else:
            self._pool = torch.nn.MaxPool1d(
                kernel_size, stride,
                padding=_pad_val,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode
            )

    def forward(self, x):
        if self._pad_mode is not None:
            x = fn.conv_pad(x, self._ksz, self._hop, self._dil, self._pad_mode)
        return self._pool(x)


class Residual1d(torch.nn.Module):

    def __init__(
        self, in_channels, out_channels,
        stride=1, downsample=None,
        batch_norm=None, weight_norm=False
    ):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = Conv1d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding="same", bias=False,
            batch_norm=batch_norm,
            batch_norm_before_act=True,
            activation="relu",
            weight_norm=weight_norm
        )
        self.conv2 = Conv1d(
            out_channels, out_channels,
            kernel_size=3, stride=1,
            padding="same", bias=False,
            batch_norm=batch_norm,
            weight_norm=weight_norm
        )
        # short cut
        if in_channels == out_channels:
            self.h = fn.Identity()
        else:
            self.h = FeatureProjection(
                in_channels, out_channels,
                bias=False, weight_norm=weight_norm
            )
        # downsample
        self.downsample = downsample
        if stride > 1:
            assert self.downsample is not None,\
                "downsample is not given for stride {}".format(stride)

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        out = x + self.h(residual)
        return out


class ResidualStack1d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, num_blocks,
        batch_norm=None, weight_norm=False,
        last_activation="relu"
    ):
        super().__init__()
        self._blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self._blocks.append(Residual1d(
                in_channels, out_channels,
                batch_norm=batch_norm,
                weight_norm=weight_norm
            ))
            in_channels = out_channels
        self._last_activation = fn.parse_activation(last_activation)

    def forward(self, x):
        for block in self._blocks:
            x = block(x)
        x = self._last_activation(x)
        return x


if __name__ == "__main__":
    conv1d   = Conv1d(16, 64, 3, 1, activation="relu", padding=1)
    deconv1d = ConvTranspose1d(64, 16, 3, 1, activation="linear", padding=1)

    x = torch.randn(2, 16, 256)
    z = conv1d(x)
    print(z.shape)
    x_ = deconv1d(z)
    print(x_.shape)

    pool = Pool1d("max", 2)
    z = pool(x)
    print(z.shape)

    residual = Residual1d(16, 64)
    z = residual(x)
    print(z.shape)

    stack = ResidualStack1d(16, 128, 3, last_activation="relu")
    z = stack(x)
    print(z.min(), z.max())
    print(z.shape)
