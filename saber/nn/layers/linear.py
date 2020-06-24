import torch
from .extend import ILayerExtended


class FullyConnected(torch.nn.Linear, ILayerExtended):

    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, bias=bias)
        self._ext_init(out_channels, batch_norm_type="1d", **kwargs)

    def forward(self, x):
        # flatten
        input_shape = x.shape
        x = x.contiguous()
        x = x.view(-1, x.size(-1))
        # run
        x = self._ext_prev_module(x)
        x = super().forward(x)
        x = self._ext_post_module(x)
        # fold
        x = x.view(*input_shape[:-1], x.size(-1))
        return x


class FeatureProjection(torch.nn.Conv1d, ILayerExtended):
    """ The feature projection layer with activation, weight initialized by glorot.
        Input's shape should be `[Batch, Feature, Time]`
    """
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super().__init__(in_channels, out_channels, 1, bias=bias)
        self._ext_init(out_channels, batch_norm_type="1d", **kwargs)

    def forward(self, x):
        assert x.dim() == 3
        x = self._ext_prev_module(x)
        x = super().forward(x)
        x = self._ext_post_module(x)
        return x
