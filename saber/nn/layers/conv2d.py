import torch
from .extend import ILayerExtended
from .. import functions as fn


class Conv2d(torch.nn.Conv2d, ILayerExtended):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding="same", dilation=1, groups=1, bias=True, **kwargs
    ):
        _pad_val, _pad_mode = fn.check_padding(padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding=_pad_val, dilation=dilation, groups=groups, bias=bias
        )
        # init extension
        self._ext_init(out_channels, batch_norm_type="2d", **kwargs)
        self._pad_mode = _pad_mode

    def forward(self, x):
        x = self._ext_prev_module(x)
        if self._pad_mode is not None:
            x = fn.conv_pad(x, self.kernel_size, self.stride,
                            self.dilation, self._pad_mode)
        x = super().forward(x)
        x = self._ext_post_module(x)
        return x


class ConvTranspose2d(torch.nn.ConvTranspose2d, ILayerExtended):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1,
        padding="same", output_padding=0, dilation=1, groups=1,
        bias=True, want_size=None, **kwargs
    ):
        _pad_val, _pad_mode = fn.check_padding(padding)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=_pad_val, output_padding=output_padding,
                         groups=groups, bias=bias, dilation=dilation)
        self._ext_init(out_channels, batch_norm_type="2d", **kwargs)
        if isinstance(_pad_mode, str):
            assert _pad_mode in ["same", "valid", "causal"]
            assert ((want_size is None) or
                    (isinstance(want_size, (list, tuple)) and
                    len(want_size) == 2)),\
                "'want_size' should be list or tuple of length 2."
            self._want_size = want_size
            self._pad_mode = _pad_mode
        else:
            self._want_size = None

    def forward(self, x):
        x = self._ext_prev_module(x)
        x = super().forward(x)
        if self._want_size is not None:
            x = fn.conv_unpad(x, self._want_size, self.kernel_size, self.stride,
                              self.dilation, self._pad_mode)
        x = self._ext_post_module(x)
        return x


class Pool2d(torch.nn.Module):
    def __init__(
        self, mode, kernel_size, stride=None, padding="same", ceil_mode=False,
        dilation=1, return_indices=False,  # max specific
        count_include_pad=True,  # pool specific
        **kwargs
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
            self._pool = torch.nn.AvgPool2d(
                kernel_size, stride,
                padding=_pad_val,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad
            )
        else:
            self._pool = torch.nn.MaxPool2d(
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


if __name__ == "__main__":
    conv   = Conv2d(16, 64, 3, 2, activation="relu")
    deconv = ConvTranspose2d(64, 16, 3, 2, activation="linear", want_size=[256, 256])

    x = torch.randn(2, 16, 256, 256)
    z = conv(x)
    print(z.shape)
    x_ = deconv(z)
    print(x_.shape)

    pool = Pool2d("max", (2, 2))
    z = pool(x)
    print(z.shape)
