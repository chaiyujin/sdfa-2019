import librosa
import webrtcvad
import numpy as np


def detect_speech(signal, sr, pad_mode='constant', smooth_ms=None, vad_mode=3):
    # init a vad
    assert 0 <= vad_mode <= 3
    vad = webrtcvad.Vad(vad_mode)  # 0~3, 3 is most aggresive mode

    # store original length
    original_length = len(signal)

    # pad and resample
    win_len = int(0.02 * sr)
    hop_len = int(0.02 * sr)
    to_pad = (win_len - hop_len) // 2
    signal = np.pad(signal, [to_pad, to_pad], pad_mode)

    # framing
    frames = [
        np.copy(signal[l: l + win_len])
        for l in range(0, len(signal) - win_len, hop_len)
    ]

    # detect
    is_speech = []
    for frame in frames:
        frame = (frame * 32767.0).astype(np.int16)
        is_speech.append(vad.is_speech(frame.tobytes(), sr))
    is_speech = np.asarray(is_speech, np.uint8)

    # smoothing
    if smooth_ms is not None:
        threshold = smooth_ms / 2.5  # (smooth_ms * sr / 1000) / hop_len
        i, last, ret = 0, 0, []
        while i < len(is_speech):
            j = i
            while j < len(is_speech) and is_speech[i] == is_speech[j]:
                j += 1
            cur = is_speech[i]
            if j - i < threshold:
                cur = last
            last = cur
            for k in range(i, j):
                ret.append(cur)
            i = j
        ret = np.asarray(ret, np.uint8)
    else:
        ret = is_speech

    # expand into original length
    ret = np.repeat(ret, repeats=hop_len)
    if original_length > len(ret):
        ret = np.pad(ret, [[0, original_length-len(ret)]], "constant", constant_values=ret[-1])

    return ret.astype(np.uint8)


def to_pairs(vad):
    pairs = []
    i = 0
    while i < len(vad):
        while i < len(vad) and vad[i] == 0:
            i += 1
        if i >= len(vad):
            break
        j = i + 1
        while j < len(vad) and vad[j] == 1:
            j += 1
        pairs.append((i, j))
        i = j
    return pairs


def from_pairs(pairs, length):
    vad = np.zeros((length), np.uint8)
    for (l, r) in pairs:
        vad[l:r] = 1
    return vad
