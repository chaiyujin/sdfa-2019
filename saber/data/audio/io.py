import os
import torch
import librosa
import numpy as np
import soundfile as sf


# load wav signal with configured sample rate
def load(path, sr, as_tensor=None):
    wav, _ = librosa.load(path, sr=sr)
    wav = wav.astype(np.float32)
    if as_tensor is not None:
        wav = torch.FloatTensor(wav).to(as_tensor)
    return wav


# save wav signal
def save(path, signal, sr):
    if torch.is_tensor(signal):
        signal = signal.detach().cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, signal, samplerate=sr)
