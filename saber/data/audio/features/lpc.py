import math
import torch
import pysptk
import numpy as np
from . import misc


def lpc_frame(frame, order, win_fn='hamm'):
    # convert into np.ndarry
    device = frame.device if torch.is_tensor(frame) else None
    if device is not None:
        frame = frame.detach().cpu().numpy()
    # remove DC compoment
    frame = frame - frame.mean()
    # window function
    if win_fn is not None:
        frame = frame * misc.get_window(win_fn, len(frame))
    # try to calculate lpc
    if np.abs(frame).max() <= 1e-4:
        feat = np.zeros(order + 1, dtype=np.float32)
    else:
        try:
            feat = pysptk.lpc(frame, order=order, use_scipy=False)
        except Exception:
            feat = np.zeros(order + 1, dtype=np.float32)
    feat[0] = 1
    feat = feat.astype(np.float32)
    # convert to tensor if necessary
    if device is not None:
        feat = torch.FloatTensor(feat).to(device)
    return feat


def lpc(signal, sr, win_size, hop_size, order, win_fn='hamm', preemphasis=0):
    signal = misc.preemphasis(signal, preemphasis)
    # convert into np.ndarry
    device = signal.device if torch.is_tensor(signal) else None
    if device is not None:
        signal = signal.detach().cpu().numpy()
    # get framed lpc
    _frames = misc.get_frames(signal, win_size, hop_size)
    _lpc_list = np.asarray([
        lpc_frame(frame, order, win_fn)
        for frame in _frames
    ]).transpose(1, 0)
    # convert to tensor if necessary
    if device is not None:
        _lpc_list = torch.FloatTensor(_lpc_list).to(device)
    return _lpc_list


class LPCLayer(torch.nn.Module):
    """ This layer doesn't support autograd """

    def __init__(self, win_size, hop_size, order, win_fn='hamm'):
        super().__init__()
        self.win_size = win_size
        self.hop_size = hop_size
        self.order = order
        self.win_fn = win_fn

    def forward(self, signals):
        # signal is store in batch
        device = signals.device
        output = np.asarray([
            lpc(
                signal   = signal.cpu().numpy(),
                win_size = self.win_size,
                hop_size = self.hop_size,
                order    = self.order,
                win_fn   = self.win_fn
            )
            for signal in signals
        ])
        return torch.FloatTensor(output).to(device)
