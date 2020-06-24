import torch
import numpy as np


class SummaryHelper(object):

    def scalar(self, tag, scalar_or_dict, global_step=None):
        """ `scalar_or_dict` is scalar or nested dict """
        if global_step is None:
            global_step = self.global_step

        def _scalar_level(tag, val):
            if isinstance(val, dict):  # dict
                for k in val:
                    _scalar_level(tag + "/" + k, val[k])
            elif val is not None:
                self.add_scalar(tag, val, global_step=global_step)

        _scalar_level(tag, scalar_or_dict)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None):
        if global_step is None:
            global_step = self.global_step
        if torch.is_tensor(img_tensor):
            img_tensor = img_tensor.detach().cpu().numpy()
        # make sure the order of dims
        img_chw = (img_tensor if img_tensor.shape[0] in [1, 3, 4]
                   else np.transpose(img_tensor, (2, 0, 1)))
        self.summary.add_image(
            tag, img_chw,
            global_step=global_step,
            walltime=walltime,
            dataformats='CHW'
        )

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if global_step is None:
            global_step = self.global_step
        self.summary.add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime)

    def add_audio(self, tag, signal, global_step=None, sample_rate=None):
        if global_step is None:
            global_step = self.global_step
        if sample_rate is None:
            sample_rate = self.hparams.audio.sample_rate
        self.summary.add_audio(tag, signal, global_step=global_step, sample_rate=sample_rate)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        if global_step is None:
            global_step = self.global_step
        self.summary.add_text(tag, text_string, global_step, walltime)

    def add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None):
        if global_step is None:
            global_step = self.global_step
        if faces is not None:
            faces = faces.astype(np.int32)
        self.summary.add_mesh(
            tag, vertices, colors,
            faces, config_dict,
            global_step, walltime
        )
