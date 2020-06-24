import saber
import numpy as np
from .preload import preload_voca as preload
from . import config
from . import non_face, nose


def set_hparams(src_hparams: saber.ConfigDict):
    config.hparams.overwrite_by(src_hparams)


def set_template(template: np.ndarray):
    template = template.flatten()
    assert len(template) == 15069
    config.hparams.set_key("template", template)


def get_template():
    return config.hparams.template


def get_indices():
    return config.hparams.tri_indices


def get_speaker_alias(speaker):
    assert speaker in config.speaker_alias_dict
    return config.speaker_alias_dict.get(speaker)
