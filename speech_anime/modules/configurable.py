import torch
import saber
from .. import layers


class Configurable(torch.nn.Module):

    def __init__(self, hparams, tag):
        super().__init__()
        self._tag = tag
        self._layers, self._parsers = layers.build_layers(
            self._tag,
            hparams.model.get(self._tag).get("layers"),
            hparams
        )

    def forward(self, x, **kwargs):
        x = layers.forward(
            self._tag, x,
            layers=self._layers,
            parsers=self._parsers,
            training=self.training,
            **kwargs
        )
        return x
