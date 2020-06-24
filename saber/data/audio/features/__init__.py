import sys
import torch
import numpy as np
from .lpc import lpc
from .spectrogram import (
    spectrogram,
    inv_spectrogram,
    mel_spectrogram,
    inv_mel_spectrogram
)
from .others import deepspeech_spec

__support__ = {
    "lpc":             "lpc",
    "mel":             "mel_spectrogram",
    "mag":             "spectrogram",
    "spec":            "spectrogram",
    "linear":          "spectrogram",
    "spectrogram":     "spectrogram",
    "deepspeech_spec": "deepspeech_spec"
}


def get(name, signal, sr, *args, **kwargs):
    assert name in __support__, "'{}' is not support!".format(name)
    # get args dict
    args_from_name = (len(kwargs) >= 1) and (len(args) == 0)
    args_from_dict = (len(kwargs) == 0) and (len(args) == 1)
    if not (args_from_name ^ args_from_dict):
        raise ValueError("saber.features.get(name, *args, **kwarg) only accept one dict or named args")
    arg_dict = args[0] if args_from_dict else kwargs
    # check sample rate
    assert arg_dict.get("sr", sr) == sr, "given two different sr: {}, {}".format(arg_dict["sr"], sr)
    arg_dict["sr"] = sr
    # get function
    fn = getattr(sys.modules[__name__], __support__[name])
    return fn(signal, **arg_dict)


def size(name, *args, **kwargs):
    assert name in __support__, "'{}' is not support!".format(name)
    # get args dict
    args_from_name = (len(kwargs) >= 1) and (len(args) == 0)
    args_from_dict = (len(kwargs) == 0) and (len(args) == 1)
    if not (args_from_name ^ args_from_dict):
        raise ValueError("saber.features.get(name, *args, **kwarg) only accept one dict or named args")
    arg_dict = args[0] if args_from_dict else kwargs
    # parse
    fn_name = __support__[name]
    if fn_name == "lpc":
        assert "order" in arg_dict
        return (arg_dict["order"] + 1)
    elif fn_name == "spectrogram":
        assert "win_size" in arg_dict
        return (arg_dict["win_size"] // 2 + 1)
    elif fn_name == "mel_spectrogram":
        return (arg_dict["n_mels"]) if "n_mels" in arg_dict else (80)
    elif fn_name == "deepspeech_spec":
        assert "win_size" in arg_dict
        return (arg_dict["win_size"] // 2 + 1)
    else:
        raise NotImplementedError()


def get_dict(name_list, signal, args_dict):
    from saber.utils import ConfigDict
    assert isinstance(name_list, (tuple, list)), "features.get_dict() want a list of names!"
    args_dict = ConfigDict(args_dict)
    args_dict.check_keys(*name_list)
    ret_dict = dict()
    for name in name_list:
        ret_dict[name] = get(name, signal, args_dict.sample_rate, **(args_dict.get(name)))
    return ret_dict
