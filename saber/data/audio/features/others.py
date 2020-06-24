import torch
import librosa
import numpy as np


def deepspeech_spec(signal, sr, win_size, hop_size, win_fn="hann", padding=False, normalize=False, preemphasis=0, eps=1e-5):
    n_fft = win_size
    win_length = n_fft
    hop_length = hop_size
    # STFT
    D = librosa.stft(
        signal, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=win_fn,
        center=padding
    )
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    if normalize:
        raise NotImplementedError()
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        if std > eps:
            spect.div_(std)

    return spect
