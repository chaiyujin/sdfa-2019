# API:
# 1. mulaw
# 2. inv_mulaw
# 3. quantize
# 4. normalize
import torch
import numpy as np


# From https://github.com/r9y9/nnmnkwii/blob/master/nnmnkwii/preprocessing/generic.py
def mulaw(y, nb_mu):
    """Mu-Law companding
    Method described in paper [1]_.
    .. math::
        f(x) = sign(x) \ln (1 + \mu |x|) / \ln (1 + \mu)
    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Compressed signal ([-1, 1])
    See also:
    .. [1] Brokish, Charles W., and Michele Lewis. "A-law and mu-law companding
        implementations using the tms320c54x." SPRA163 (1997).
    """
    mu = float(nb_mu)
    return _sign(y) * _log1p(_abs(y) * mu) / _log1p(mu)


# From https://github.com/r9y9/nnmnkwii/blob/master/nnmnkwii/preprocessing/generic.py
def inv_mulaw(y, nb_mu):
    """Inverse of mu-law companding (mu-law expansion)
    .. math::
        f^{-1}(x) = sign(y) (1 / \mu) (1 + \mu)^{|y|} - 1)
    Args:
        y (array-like): Compressed signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.
    Returns:
        array-like: Uncomprresed signal (-1 <= x <= 1)
    """
    mu = float(nb_mu)
    return _sign(y) * (1.0 / mu) * ((1.0 + mu)**_abs(y) - 1.0)


def quantize(y, nb_mu):
    return _long((y + 1.0) * float(nb_mu) / 2.0)


def normalize(y, nb_mu):
    return _float(y) * 2.0 / float(nb_mu) - 1.0


# ------------- #
# tools of math #
# ------------- #

def _float(y):
    if torch.is_tensor(y):
        return y.float()
    elif isinstance(y, np.ndarray):
        return y.astype(np.float32)
    else:
        return float(y)


def _long(y):
    if torch.is_tensor(y):
        return y.long()
    elif isinstance(y, np.ndarray):
        return y.astype(np.int64)
    else:
        return int(y)


def _sign(y):
    return y.sign() if torch.is_tensor(y) else np.sign(y)


def _log1p(y):
    return y.log1p() if torch.is_tensor(y) else np.log1p(y)


def _abs(y):
    return y.abs() if torch.is_tensor(y) else np.abs(y)
