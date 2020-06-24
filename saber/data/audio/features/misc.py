import math
import torch
import librosa
import numpy as np
from functools import lru_cache


def preemphasis(signal, a=0):
    if a is None or a == 0:
        return signal
    # assert 0 < a < 1, "preemphasis should be in (0, 1), but '{}' is given".format(a)
    if torch.is_tensor(signal):
        assert signal.dim() == 1
        return torch.cat((signal[0: 1], signal[1:] - signal[:-1] * a))
    elif isinstance(signal, np.ndarray):
        assert signal.ndim == 1
        return np.append(signal[0], signal[1:] - a * signal[:-1])
    else:
        raise TypeError("unknow signal type: {}".format(type(signal)))


def deemphasis(signal, a=0):
    if a is None or a == 0:
        return signal
    # assert 0 < a < 1, "deemphasis should be in (0, 1), but '{}' is given".format(a)
    for i in range(1, len(signal)):
        signal[i] += signal[i-1] * a


def get_window(win_fn, win_size, device=None):
    if device is not None:
        return __get_window_torch(win_fn, win_size, device)
    else:
        return __get_window_numpy(win_fn, win_size)


def get_mel_filters(sr, n_fft, n_mels, fmin, fmax, device=None):
    if device is not None:
        return __get_mel_filters_torch(sr, n_fft, n_mels, fmin, fmax, device)
    else:
        return __get_mel_filters_numpy(sr, n_fft, n_mels, fmin, fmax)


def get_inv_mel_filters(sr, n_fft, n_mels, fmin, fmax, device=None):
    if device is not None:
        return __get_inv_mel_filters_torch(sr, n_fft, n_mels, fmin, fmax, device)
    else:
        return __get_inv_mel_filters_numpy(sr, n_fft, n_mels, fmin, fmax)


def get_frames(signal, win_size, hop_size, win_fn=None):
    """
    Segment a signal into overlapping frames.
    Parameters:
        signal: [length], np.ndarray or torch.Tensor
        win_size, hop_size: int
        win_fn: str, see `get_window()`
    Return:
    -------
        frames: [num_frames, win_size], framed signal ndarray.
    """
    slen = len(signal)
    if slen < win_size:
        numframes = 1
        if torch.is_tensor(signal):
            signal = torch.cat((signal, torch.zeros((win_size-slen), dtype=signal.dtype).to(signal.device)))
        else:
            signal = np.pad(signal, [[0, win_size - slen]], "constant")
    else:
        numframes = 1 + int(math.floor((1.0 * slen - win_size)/hop_size))

    # index select
    if torch.is_tensor(signal):
        # for tensor
        sizes = (numframes, win_size)
        strides = (hop_size * signal.stride(0), signal.stride(0))
        frames = signal.as_strided(sizes, strides)
    else:
        # for numpy
        indices = (np.tile(np.arange(0, win_size), (numframes, 1)) +
                   np.tile(np.arange(0, numframes * hop_size, hop_size),
                           (win_size, 1)).T)
        indices = np.array(indices, dtype=np.int32)
        frames = signal[indices]
    # window
    if win_fn is not None:
        if torch.is_tensor(signal):
            frames *= __get_window_torch(win_fn, win_size, signal.device).unsqueeze(0)
        else:
            frames *= np.expand_dims(__get_window_numpy(win_fn, win_size), axis=0)
    return frames


@lru_cache(maxsize=None, typed=True)
def __get_window_numpy(win_fn, win_size):
    # check win_fn
    assert win_fn in ["hamm", "hann", "ones", "hamming", "hanning"]
    if win_fn in ["hamm", "hann"]:
        win_fn += "ing"
    return getattr(np, win_fn)(win_size).astype(np.float32)


@lru_cache(maxsize=None, typed=True)
def __get_window_torch(win_fn, win_size, device):
    _win = __get_window_numpy(win_fn, win_size)
    return torch.FloatTensor(_win).to(device)


@lru_cache(maxsize=None, typed=True)
def __get_mel_filters_numpy(sr, n_fft, n_mels, fmin, fmax):
    return librosa.filters.mel(
        sr     = sr,
        n_fft  = n_fft,
        n_mels = n_mels,
        fmin   = fmin,
        fmax   = fmax
    ).astype(np.float32)


@lru_cache(maxsize=None, typed=True)
def __get_mel_filters_torch(sr, n_fft, n_mels, fmin, fmax, device):
    _filters = __get_mel_filters_numpy(sr, n_fft, n_mels, fmin, fmax)
    return torch.FloatTensor(_filters).to(device)


@lru_cache(maxsize=None, typed=True)
def __get_inv_mel_filters_numpy(sr, n_fft, n_mels, fmin, fmax):
    return np.linalg.pinv(librosa.filters.mel(
        sr     = sr,
        n_fft  = n_fft,
        n_mels = n_mels,
        fmin   = fmin,
        fmax   = fmax
    )).astype(np.float32)


@lru_cache(maxsize=None, typed=True)
def __get_inv_mel_filters_torch(sr, n_fft, n_mels, fmin, fmax, device):
    _inv_filters = __get_inv_mel_filters_numpy(sr, n_fft, n_mels, fmin, fmax)
    return torch.FloatTensor(_inv_filters).to(device)
