""" for rms normalize """
import os
import librosa
import numpy as np
from tqdm import tqdm
from saber.utils import log


def analyze_db_dataset(wav_root, sr, silence_threshold=-40,
                       max_amplitude=0.999, target_db=-16):
    # find all wavs
    wavpath_list = []
    for root, _, files in os.walk(wav_root):
        for filename in files:
            if os.path.splitext(filename)[1] == ".wav":
                filename = os.path.join(root, filename)
                wavpath_list.append(os.path.abspath(filename))
    assert max_amplitude < 1
    top_db = 20.0 * np.log10(max_amplitude)
    db_tuples = {}
    recommend_db = target_db or 0
    progress = tqdm(wavpath_list)
    for wavpath in progress:
        wav = librosa.core.load(wavpath, sr=sr)[0]
        rms_db, max_db = analyze_db(wav, threshold=silence_threshold)
        if rms_db is None:
            continue
        db_tuples[wavpath] = {
            "rms_db": rms_db,
            "max_db": max_db
        }
        # update recommend_db
        delta_db = recommend_db - rms_db
        if max_db + delta_db > top_db:
            recommend_db = top_db - max_db + rms_db
        progress.set_description("recommend {:.2f}dB".format(recommend_db))
    # ceil recommend
    recommend_db = np.floor(recommend_db * 100.0) / 100.0
    log.info("recommend set target_db as {} (max amplitude <= {})".format(
        recommend_db, max_amplitude
    ))
    return recommend_db, db_tuples


def analyze_db(wav, threshold=None):
    # get maximum
    db = 20.0 * np.log10(np.maximum(np.abs(wav), 1e-10))
    max_db = db.max()
    # mask silence dynamicly
    # threshold += max(max_db, -6.0)  # at least 0.5
    if threshold is None:
        threshold = db.min()
    mask = db >= threshold
    # all silence
    if mask.sum() == 0:
        return None, None
    # get mean db
    rms = np.sqrt(np.mean(wav[mask] ** 2))
    rms_db = 20.0 * np.log10(rms)
    return rms_db, max_db


def normalize(wav, target_db=-20, threshold=None, rms_db=None, max_db=None):
    if rms_db is not None:
        assert max_db is not None
    else:
        rms_db, max_db = analyze_db(wav, threshold=threshold)
    if rms_db is None:
        # all silence
        return wav
    # delta
    delta_db = target_db - rms_db
    if delta_db + max_db > 0:
        log.warn("[rms]: max db {:.2f} will > 0,"
                 "signal will be clipped".format(max_db + delta_db))
    scale_rms = np.power(10.0, delta_db / 20.0)
    rms_wav = wav * scale_rms
    return np.clip(rms_wav, -0.999, 0.999)
