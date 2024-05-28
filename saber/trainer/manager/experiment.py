import os
import torch
import saber
import inspect
import numpy as np
from saber.utils import log, ConfigDict
from torch.utils.tensorboard import SummaryWriter
# saber model
from ..saber_model import SaberModel
# basic extensions
from .loss_info import LossInformation
from .checkpoints import CheckpointIO
from .device_mover import DeviceMover
from .tb_helpers import SummaryHelper


class Experiment(CheckpointIO, DeviceMover, LossInformation, SummaryHelper):
    __plot_functions__ = []

    def __init__(
        self,
        model,
        hparams,
        log_dir,
        training         = True,
        # data loaders for trainer
        train_loaders    = dict(),
        valid_loaders    = dict(),
        main_loader_name = None,
        # log information for trainer
        nb_train_log     = 2,
        hp_dump_name     = "hparams.json",
        loss_subdir      = "train_log/loss",
        video_subdir     = "train_log/video",
        image_subdir     = "train_log/image",
        audio_subdir     = "train_log/audio",
        ckpts_subdir     = "checkpoints",
        # Checkpoint preprocess callback
        ckpt_preprocess  = None,
    ):
        # check args
        self._check_args(hparams, model, train_loaders, valid_loaders)

        # model size
        total_params = sum(p.numel() for p in model.parameters())
        params_info = (
            f"Model has {total_params:,} parameters. "
            f"Total size is {total_params * 4 / 1024 / 1024:.2f}MB (assume as float32)."
        )
        saber.log.info(params_info)

        # exp memebrs
        self._log_dir = log_dir
        self._num_log = nb_train_log
        self._loss_dir = os.path.join(log_dir, loss_subdir)
        self._video_dir = os.path.join(log_dir, video_subdir)
        self._image_dir = os.path.join(log_dir, image_subdir)
        self._audio_dir = os.path.join(log_dir, audio_subdir)
        self._ckpts_dir = os.path.join(log_dir, ckpts_subdir)
        self._training = training
        # model, hparams
        self.hparams = hparams
        self.model = model

        # data loaders
        self._init_loaders(train_loaders, valid_loaders, main_loader_name)

        # training or not
        if training:
            # make dirs
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(self._loss_dir, exist_ok=True)
            os.makedirs(self._video_dir, exist_ok=True)
            os.makedirs(self._image_dir, exist_ok=True)
            os.makedirs(self._audio_dir, exist_ok=True)
            os.makedirs(self._ckpts_dir, exist_ok=True)
            # dump hparams
            self.hparams.dump(filename=hp_dump_name, dump_dir=self.log_dir)
            # get configured optimizers
            self.optimizers = self.model.configure_optimizers()
            # tensorboard summary
            self.summary = SummaryWriter(log_dir)
            # dump parameters information
            with open(os.path.join(log_dir, "params_info.txt"), "w") as fp:
                fp.write(params_info)
            # model mode
            self.model.train()
        else:
            self.optimizers = None
            self.summary = None
            # model mode
            self.model.eval()

        # load ckpt
        if self.hparams.get("load_from") is not None:
            self.load_checkpoint(self.hparams.load_from, preprocess=ckpt_preprocess)

        # move device
        self.move_to(self.hparams.device)

        # cache index
        self._global_step = None
        self._current_epoch = None

    # -------------- #
    #  init helpers  #
    # -------------- #

    def _check_args(self, hparams, model, train_loaders, valid_loaders):
        assert type(hparams) is ConfigDict,\
            "[exp]: Given 'hparams' is not {}".format(type(ConfigDict))
        assert isinstance(model, SaberModel),\
            "[exp]: Given 'model' is not {}".format(type(SaberModel))
        for name, loader in train_loaders.items():
            assert isinstance(loader, torch.utils.data.dataloader.DataLoader),\
                f"[exp]: Given '{name}' in train_loaders is not torch DataLoader."
        for name, loader in valid_loaders.items():
            assert isinstance(loader, torch.utils.data.dataloader.DataLoader),\
                f"[exp]: Given '{name}' in valid_loaders is not torch DataLoader."

    def _init_loaders(self, train_loaders, valid_loaders, main_loader_name):
        self.train_loaders = train_loaders
        self.valid_loaders = valid_loaders
        self.train_iterators = dict()
        self.valid_iterators = dict()
        self.main_train_loader = None
        self.main_valid_loader = None
        self.main_loader_name = main_loader_name
        # find main loader in given dict
        if len(self.train_loaders) > 0:
            if len(self.train_loaders) == 1 and self.main_loader_name is None:
                self.main_loader_name = list(self.train_loaders.keys())[0]
            assert self.main_loader_name is not None,\
                "main_loader_name is not given!"
            assert self.main_loader_name in self.train_loaders,\
                f"failed to find '{self.main_loader_name}' in train_loaders"
            self.main_train_loader = self.train_loaders[self.main_loader_name]
        if len(self.valid_loaders) > 0:
            if len(self.valid_loaders) == 1 and self.main_loader_name is None:
                self.main_loader_name = list(self.valid_loaders.keys())[0]
            assert self.main_loader_name is not None,\
                "main_loader_name is not given!"
            assert self.main_loader_name in self.valid_loaders,\
                f"failed to find '{self.main_loader_name}' in valid_loaders"
            self.main_valid_loader = self.valid_loaders[self.main_loader_name]
        # convert other loaders to iter
        for name in self.train_loaders:
            if name != self.main_loader_name:
                self.train_iterators[name] = iter(self.train_loaders[name])
        for name in self.valid_loaders:
            if name != self.main_loader_name:
                self.valid_iterators[name] = iter(self.valid_loaders[name])

    # ---------------------- #
    #  custom plot function  #
    # ---------------------- #

    def plot_nested(self, tag, preds, batch, global_step=None):
        """ plot nested preds and given training batch """
        if global_step is None:
            global_step = self.global_step

        def _plot_level(tag, pred_dict, level):
            for fn in Experiment.__plot_functions__:
                fn(
                    exp=self,
                    num=self._num_log,
                    tag=tag,
                    preds=pred_dict,
                    batch=batch
                )
            for k in pred_dict:
                if isinstance(pred_dict[k], dict):
                    _plot_level("{}/{}".format(tag, k), pred_dict[k], level + 1)

        _plot_level(tag, preds, 0)

    @classmethod
    def register_plot(cls, fn):
        if fn not in cls.__plot_functions__:
            params = [k for k in inspect.signature(fn).parameters]
            assert params == ["exp", "num", "tag", "preds", "batch"],\
                "Experiment.register_plot() need signature: fn(exp, num, tag, preds, batch)"
            cls.__plot_functions__.append(fn)
        return fn

    # ---------------------- #
    #  read-only properties  #
    # ---------------------- #

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def loss_dir(self):
        return self._loss_dir

    @property
    def video_dir(self):
        return self._video_dir

    @property
    def image_dir(self):
        return self._image_dir

    @property
    def audio_dir(self):
        return self._audio_dir

    @property
    def checkpoint_dir(self):
        return self._ckpts_dir

    @property
    def training(self):
        return self._training

    @property
    def is_parallel(self):
        return (
            isinstance(self.model, torch.nn.DataParallel) or
            isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
        )

    @property
    def saber_model(self):
        # return the saber model class
        # unwrapper the parallel module
        return self.model.module if self.is_parallel else self.model

    @property
    def on_gpu(self):
        return self.saber_model.on_gpu

    # --------- #
    #  indices  #
    # --------- #

    @property
    def global_step(self):
        self._global_step = self.saber_model.global_step
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value
        self.saber_model.global_step = value

    @property
    def current_epoch(self):
        self._current_epoch = self.saber_model.current_epoch
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value):
        self._current_epoch = value
        self.saber_model.current_epoch = value
