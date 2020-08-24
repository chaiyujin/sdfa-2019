import os
import cv2
import saber
import librosa
import numpy as np


def windowed_features(
    signal,
    signal_stt,
    signal_end,
    # for feature
    audio_config,
    # framewise phonemes,
    ph_aligned=None,
    # augment args
    signal_noise=None,
    feat_extra=None,
    feat_scale=None,
    feat_noise=None,
    feat_tremolo=None,
    feat_dropout=None,
    random_args=None,
):
    if random_args is None:
        random_args = dict()
    assert isinstance(random_args, dict)

    # analyze feature args
    assert isinstance(audio_config, saber.ConfigDict), f"{type(audio_config)}"
    sr = audio_config.get("sample_rate")
    feat_config = audio_config.get("feature")
    assert isinstance(feat_config, saber.ConfigDict)

    feat_name   = feat_config.get("name")
    with_delta  = feat_config.get("with_delta")
    frames      = feat_config.get("sliding_window_frames")

    main_name = feat_name.split("-")[0]
    win_size = audio_config.get(main_name).get("win_size")
    hop_size = audio_config.get(main_name).get("hop_size")

    # ex_time
    wl, wr = signal_stt, signal_end
    ex_feat, ex_time = 0, 0
    if feat_extra is not None:
        assert isinstance(feat_extra, (list, tuple))
        assert len(feat_extra) == 2
        ex_feat, ex_time = feat_extra
    # # make sure in range
    # if ex_time > 0:
    #     ex_time = min([ex_time, wl // hop_size, (len(signal) - wr) // hop_size])
    wl -= ex_time * hop_size
    wr += ex_time * hop_size
    assert wl < wr, f"ex_time {ex_time} is to large, no more samples remain."
    # get proper signal window
    if wr <= 0 or wl >= len(signal):
        wav = np.zeros((wr-wl), np.float32)
    elif 0 <= wl and wr <= len(signal):
        wav = np.copy(signal[wl: wr])  # copy, without change original signal
    else:
        pad_wav = [0, 0]
        if wl < 0:
            pad_wav[0] = -wl
        if wr > len(signal):
            pad_wav[1] = wr - len(signal)
        wav = np.pad(signal[max(wl, 0): min(wr, len(signal))], [pad_wav], "constant")
    assert len(wav) == wr - wl
    # signal noise
    if isinstance(signal_noise, str):
        noise_type, noise_scale = signal_noise.split("@")
        noise_scale = float(noise_scale)
        assert noise_type in ["pink", "white", "none"]
        if   noise_type == "pink":  wav += saber.audio.pink_noise (wr-wl, noise_scale)
        elif noise_type == "white": wav += saber.audio.white_noise(wr-wl, noise_scale)
    elif signal_noise is not None:
        length = wr-wl

        s = random_args.get("signal_noise_start")
        if s is None: s = np.random.randint(0, len(signal_noise)-length+1)
        random_args["signal_noise_start"] = s

        e = s + length
        noise = signal_noise[s:e]
        if len(noise) < length:
            noise = np.pad(noise, [[0, length-len(noise)]], "constant")
        wav += noise

    # get feature, TODO: other features here
    feats_dict = saber.audio.features.get_dict([feat_name], wav, audio_config)

    # get phonemes
    ph_labels = None
    if ph_aligned is not None:
        # get framewise phonemes
        time_scale = (frames + ex_time * 2) / frames
        pl = signal_stt - ex_time * hop_size
        pr = pl + hop_size * time_scale * frames + (win_size - hop_size) * time_scale
        ph_labels = saber.text.PhonemeUtils.segment_phonemes(
            ph_aligned, sr, pl, pr, win_size * time_scale, hop_size * time_scale
        )

    dst_num_feats = dict()
    for name in feats_dict:
        feat = feats_dict[name]
        assert feat.ndim == 2
        dst_num_feats[name] = feat.shape[0]

        # only extend freq for following features
        if name not in ["mel", "mag", "spec", "linear", "spectrogram"]:
            continue

        # remove / append freq
        trunck = random_args.get("trunck")
        pad_mode = random_args.get("pad_mode")
        lower_freq = random_args.get("lower_freq")
        if trunck is None: trunck = (np.random.uniform() < 0.5)
        if pad_mode is None: pad_mode = np.random.choice(["reflect", "constant"])
        if lower_freq is None: lower_freq = (np.random.uniform() < 0.5)

        random_args["trunck"] = trunck
        random_args["pad_mode"] = pad_mode
        random_args["lower_freq"] = lower_freq

        if ex_feat < 0:
            if lower_freq: feat = feat[-ex_feat:]
            else:          feat = feat[:ex_feat]
        elif ex_feat > 0:
            if lower_freq:
                # append in lower frequency
                feat = np.pad(feat, [[ex_feat, 0], [0, 0]], "constant")
                # maybe directly trunck high freq
                if trunck:
                    feat = feat[:-ex_feat]
            else:
                # append in higher frequency
                feat = np.pad(feat, [[0, ex_feat], [0, 0]], pad_mode)
                # maybe directly trunck low freq
                if trunck:
                    feat = feat[ex_feat:]

        # tremolo
        if feat_tremolo is not None and feat_tremolo > 0:
            cols = list(feat.transpose(1, 0))
            shifting = np.abs(np.sin(np.linspace(0, np.pi * 2, num=len(cols)) * feat_tremolo))
            shifting = (shifting * 3.0).astype(np.int32)
            for c in range(len(cols)):
                col = cols[c]
                # pad = np.random.randint(0, 5)
                pad = shifting[c]
                if pad > 0:
                    col = np.pad(col[:-pad], [[pad, 0]], "constant")
                cols[c] = col
            feat = np.asarray(cols).transpose(1, 0)

        feats_dict[name] = feat

    # resize into target shape, maybe scale, noise, dropout
    for name in feats_dict:
        feat = np.expand_dims(feats_dict[name], axis=0)
        assert feat.shape[2] == frames + ex_time * 2
        feat = cv2.resize(
            feat.transpose(1, 2, 0), (frames, dst_num_feats[name]),
            interpolation=cv2.INTER_LINEAR
        )
        assert feat.ndim == 2
        # scale, noise, dropout
        if feat_scale is not None:
            feat *= feat_scale
            # print(feat_scale[:, 0])
        if feat_noise is not None and feat_noise > 0:
            feat += np.random.normal(0.0, feat_noise, size=feat.shape)
        if feat_dropout is not None and feat_dropout > 0:
            n_feat = feat.shape[0]
            mask_len = max(1, int(feat_dropout * n_feat))

            mask_idx = random_args.get("mask_idx")
            drop_mode = random_args.get("drop_mode")
            mask_thres = random_args.get("mask_thres")
            if mask_idx is None: mask_idx = np.random.choice(np.arange(n_feat), mask_len)
            if drop_mode is None: drop_mode = np.random.choice(["zero", "max"])
            if mask_thres is None: mask_thres = np.random.uniform(0.3, 0.6)

            random_args["mask_idx"] = mask_idx
            random_args["drop_mode"] = drop_mode
            random_args["mask_thres"] = mask_thres

            if drop_mode == "zero":
                feat[mask_idx] = 0
            else:
                where = feat[mask_idx] < mask_thres
                feat[mask_idx][where] = mask_thres
        feats_dict[name] = feat

    # delta
    if with_delta:
        # on channel dim

        def delta_fn(feat, n):
            return librosa.feature.delta(feat, order=n)

        all_feats = []
        all_delta = []
        all_delta2 = []
        delta = delta_fn(feats_dict[feat_name], 1)
        delta2 = delta_fn(feats_dict[feat_name], 2)
        all_feats.append(feats_dict[feat_name])
        all_delta.append(delta)
        all_delta2.append(delta2)
        audio_feat = np.concatenate([
            np.expand_dims(np.concatenate(all_feats,  axis=0), axis=0),
            np.expand_dims(np.concatenate(all_delta,  axis=0), axis=0),
            np.expand_dims(np.concatenate(all_delta2, axis=0), axis=0),
        ], axis=0)
    else:
        # on feature dim
        all_feats = []
        all_feats.append(feats_dict[feat_name])
        if with_delta:
            all_feats.append(feats_dict[feat_name+"-delta"])
        audio_feat = np.expand_dims(np.concatenate(all_feats, axis=0), axis=0)

    return audio_feat.astype(np.float32), ph_labels, wav, random_args
