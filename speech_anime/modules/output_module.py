import saber
import torch
import numpy as np
from .. import layers
from ..tools import PredictionType, FaceDataType


class OutputModule(torch.nn.Module):
    def __init__(self, hparams, load_pca):
        super().__init__()
        hp = hparams.model.output
        # check prediction type
        pred_type = hparams.model.prediction_type
        face_type = hparams.model.face_data_type
        assert pred_type in PredictionType.valid_types()
        assert face_type in FaceDataType.valid_types()
        self._pred_type = PredictionType[pred_type]
        self._face_type = FaceDataType[face_type]
        self._using_pca = hp.using_pca
        self._return_pca = (self._pred_type.name.find("pca") == 0)
        if self._return_pca:
            assert self._using_pca
        # build layers
        self._layers, self._parsers = layers.build_layers("output", hp.layers, hparams)
        # two branch for dgrad
        if self._face_type == FaceDataType.dgrad_3d:
            self._scale_layers, self._scale_parsers =\
                layers.build_layers("output-scale", hp.layers_scale, hparams)
            self._rotat_layers, self._rotat_parsers =\
                layers.build_layers("output-rotat", hp.layers_rotat, hparams)
            if hp.using_pca:
                self._scale_pca = PcaInversion(
                    *hp.pca_scale, trainable=hp.pca_trainable,
                    coeffs_dim=self._scale_parsers[-1]["out_channels"],
                    output_dim=hp.output_dim_scale, load_pca=load_pca
                )
                self._rotat_pca = PcaInversion(
                    *hp.pca_rotat, trainable=hp.pca_trainable,
                    coeffs_dim=self._rotat_parsers[-1]["out_channels"],
                    output_dim=hp.output_dim_rotat, load_pca=load_pca
                )
                self._scale_pca_size = self._scale_pca.compT.size(1)
        else:
            if hp.using_pca:
                self._pca = PcaInversion(
                    *hp.pca, trainable=hp.pca_trainable,
                    coeffs_dim=self._parsers[-1]["out_channels"],
                    output_dim=hp.output_dim, load_pca=load_pca
                )

    def forward(self, x, **kwargs):
        assert x.dim() == 3
        N, L, C = x.size()
        x = layers.forward(
            "output", x,
            layers=self._layers,
            parsers=self._parsers,
            training=self.training,
            **kwargs
        )
        if self._face_type == FaceDataType.dgrad_3d:
            x_scale = layers.forward(
                "output-scale", x,
                layers=self._scale_layers,
                parsers=self._scale_parsers,
                training=self.training,
                **kwargs
            )
            x_rotat = layers.forward(
                "output-rotat", x,
                layers=self._rotat_layers,
                parsers=self._rotat_parsers,
                training=self.training,
                **kwargs
            )
            # need return data
            if self._using_pca and (not self._return_pca):
                x_scale = self._scale_pca(x_scale).view(N, L, -1, 6)
                x_rotat = self._rotat_pca(x_rotat).view(N, L, -1, 3)
            return x_scale, x_rotat
            # postfix = "_pca" if self._return_pca else ""
            # return {
            #     f"dgrad_3d_scale{postfix}": x_scale,
            #     f"dgrad_3d_rotat{postfix}": x_rotat
            # }
        else:
            if self._using_pca and (not self._return_pca):
                x = self._pca(x)
            return x
            # postfix = "_pca" if self._return_pca else ""
            # return {f"{self._face_type.name}{postfix}": x}


class PcaInversion(torch.nn.Module):
    def __init__(self, pca_compT, pca_means, trainable,
                 coeffs_dim, output_dim, load_pca):
        super().__init__()
        if load_pca:
            if isinstance(pca_compT, str): pca_compT = np.load(pca_compT)
            if isinstance(pca_means, str): pca_means = np.load(pca_means)
            if isinstance(pca_compT, np.ndarray): pca_compT = pca_compT.astype(np.float32)
            if isinstance(pca_means, np.ndarray): pca_means = pca_means.astype(np.float32)
        else:
            saber.log.warn("PCA is not loaded, use zeros. Make sure load from checkpoint afterwards.")
            pca_compT = np.zeros((output_dim, coeffs_dim), dtype=np.float32)
            pca_means = np.zeros((output_dim), dtype=np.float32)

        if trainable:
            self.register_parameter("compT", torch.FloatTensor(pca_compT))
            self.register_parameter("means", torch.FloatTensor(pca_means))
        else:
            self.register_buffer("compT", torch.FloatTensor(pca_compT))
            self.register_buffer("means", torch.FloatTensor(pca_means))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.compT, self.means)
