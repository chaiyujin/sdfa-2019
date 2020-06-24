import os
import cv2
import saber
import pickle
import numpy as np
from . import data_info
from ..config import hparams


def _delta_timestamps(data):
    # ts_delta = int(hparams.anime.feature.ts_delta)
    return data


def load_npy_data(path, symmetric_upper=None, force_mask=False, with_ts_delta=True):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
        if with_ts_delta:
            data = _delta_timestamps(data)
    return data
