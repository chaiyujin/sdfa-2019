import math
import torch
import torch.nn.functional as F
from ..tools import PredictionType, FaceDataType


class PLoss(torch.nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self._fn = F.mse_loss
        self._pred_type = PredictionType[hparams.model.prediction_type]
        self._face_type = FaceDataType[hparams.model.face_data_type]

    def forward(self, inputs, targets, weights):
        if self._pred_type == PredictionType.pca_normal:
            raise NotImplementedError()
        else:
            if (
                self._face_type == FaceDataType.dgrad_3d and
                self._pred_type == PredictionType.face_data
            ):
                assert inputs.dim() == 4
                assert targets.dim() == 4
                if inputs.size(-1) == 3:
                    inputs = torch.exp(inputs)
                    targets = torch.exp(targets)
            loss = self._fn(inputs, targets, reduction='none')

        # reduce into batch size
        if self._face_type == FaceDataType.dgrad_3d:
            loss = loss.sum(-1)  # ! because scale, rotat has different size
        while loss.dim() > 1:
            loss = loss.mean(-1)

        return (loss * weights).mean(dim=0)


class MLoss(torch.nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self._fn = F.mse_loss
        self._pred_type = PredictionType[hparams.model.prediction_type]
        self._face_type = FaceDataType[hparams.model.face_data_type]

    def forward(self, inputs, targets, weights):
        bhs = inputs.shape[0] // 2
        if self._pred_type == PredictionType.pca_normal:
            raise NotImplementedError()
        else:
            if (
                self._face_type == FaceDataType.dgrad_3d and
                self._pred_type == PredictionType.face_data
            ):
                assert inputs.dim() == 4
                assert targets.dim() == 4
                if inputs.size(-1) == 3:
                    inputs = torch.exp(inputs)
                    targets = torch.exp(targets)
            # get loss
            m_pred = inputs[bhs:] - inputs[:bhs]
            m_true = targets[bhs:] - targets[:bhs]
            loss = self._fn(m_pred, m_true, reduction='none')
        half_weights = (weights[bhs:] + weights[:bhs])

        # reduce into batch size
        if self._face_type == FaceDataType.dgrad_3d:
            loss = loss.sum(-1)
        while loss.dim() > 1:
            loss = loss.mean(-1)

        return (loss * half_weights).mean(dim=0)


class ELoss(torch.nn.Module):
    def __init__(self, hparams, eps=1e-10, vmax=1e-2):
        super().__init__()
        self.eps = eps
        self.max = vmax
        self._fn = F.mse_loss

    def forward(self, inputs):
        bhs = int(inputs.size(0)) // 2
        loss = ((inputs[bhs:] - inputs[:bhs]) ** 2)
        magn = inputs ** 2
        return loss.sum(dim=1) * 2 / magn.mean()


class DynamicLossScaler(object):
    def __init__(self, beta=0.99, eps=1e-8):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reset_state()

    def reset_state(self):
        self.vt = 0.0
        self.beta_t = 1.0
        self.scale = 1.0

    def scale_loss(self, loss, training):
        if training:
            # assert loss.dim() == 1
            loss_ms = float((loss ** 2).mean(dim=0).detach())
            # update scale
            self.beta_t = self.beta_t * self.beta
            self.vt = self.beta * self.vt + (1.0 - self.beta) * loss_ms
            self.scale = math.sqrt(self.vt / (1.0 - self.beta_t)) + self.eps
        # backward
        scaled_loss = loss.mean() / self.scale
        return scaled_loss
