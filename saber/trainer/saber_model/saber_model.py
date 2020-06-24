"""Adapt from pytorch_lightning LightningModule
"""

import torch
import saber
from .memory import ModelSummary
from .grads import GradInformation


class SaberModel(GradInformation):

    __step_should_return__ = (
        "SaberModel.{}_step(self, batch, i_batch) should return:"
        "  pred_dict, loss_dict[, scalar_dict]"
    )

    def __init__(self, hparams, trainset, validset):
        super().__init__()
        self.hp = hparams
        # device indicators
        self._on_gpu = False
        # training indices
        self.global_step = 0
        self.current_epoch = 0
        # datasets
        self.trainset = trainset
        self.validset = validset
        # example inputs
        self.example_input_array = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train_step(self, batch, i_batch):
        # must be implemented for trainer
        raise NotImplementedError()

    def valid_step(self, batch, i_batch):
        # optional for trainer
        pass

    def evaluate(self, sources, experiment=None, in_trainer=True):
        # optional for trainer
        pass

    def configure_optimizers(self):
        hp = self.hp.optim
        hp.check_keys("name", "args")
        hp.args.check_keys("lr")
        optim = getattr(torch.optim, hp.name)(self.parameters(), **hp.args)
        optim_sch = None
        if hp.get("lr_scheduler") is not None:
            hp.lr_scheduler.check_keys("name", "args")
            optim_sch = getattr(saber.trainer.lr_schedulers, hp.lr_scheduler.name)(
                optim, **hp.lr_scheduler.args
            )
        return {hp.name: (optim, optim_sch)}

    def summarize(self):
        model_summary = ModelSummary(self)
        saber.log.info("summary\n{}".format(model_summary))

    def has_implemented(self, fn_name):
        return fn_name in self.__class__.__dict__

    @property
    def on_gpu(self):
        return self._on_gpu
