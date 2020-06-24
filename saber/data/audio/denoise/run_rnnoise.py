import os
import librosa
import numpy as np
from saber.utils import log

_git_path = os.path.join(os.path.dirname(__file__), "rnnoise/")
_git_url = "https://github.com/chaiyujin/rnnoise.git"
_bin_path = os.path.join(_git_path, "examples/rnnoise_demo")


def _make():
    cur_path = os.getcwd()
    # begin to make
    os.chdir(_git_path)
    os.system("./autogen.sh && ./configure && make")
    # exit make
    os.chdir(cur_path)


def run_rnnoise_demo(wav, sr, specific_key=""):
    """ 'specific_key' is used to handle multi-thread """

    import matplotlib.pyplot as plt
    tmp_noise = os.path.join(_git_path, f"tmp_input_{specific_key}.pcm")
    tmp_after = os.path.join(_git_path, f"tmp_output_{specific_key}.pcm")

    resampled = librosa.resample(wav, sr, 48000)
    resampled = (resampled * 32767.0).astype(np.int16)
    resampled = np.pad(resampled, [[0, 1024]], "constant")
    resampled.tofile(tmp_noise)
    ret = os.system("{} {} {}".format(_bin_path, tmp_noise, tmp_after))
    if ret != 0:
        raise RuntimeError("Error when denoise.")
    denoised = np.fromfile(tmp_after, dtype=np.int16)
    # adjust length
    real_length = len(resampled) - 1024
    denoised = denoised[:real_length]
    if len(denoised) < real_length:
        denoised = np.pad(denoised, [[0, real_length - len(denoised)]],
                          "constant")
    # print(len(resampled))
    # print(len(denoised), real_length)
    # plt.plot(resampled[351000*3:real_length], label="original")
    # plt.plot(denoised[351000*3:], label="denoised")
    # plt.legend()
    # plt.show()
    # quit()
    denoised = (denoised.astype(np.float32) / 32767.0)
    denoised = librosa.resample(denoised, 48000, sr)

    os.remove(tmp_noise)
    os.remove(tmp_after)
    return denoised


# check and auto clone, make
if not os.path.exists(_git_path):
    log.warn("Failed to find 'rnnoise', git clone.")
    os.system("git clone {} {}".format(_git_url, _git_path))
if not os.path.exists(_bin_path):
    _make()
