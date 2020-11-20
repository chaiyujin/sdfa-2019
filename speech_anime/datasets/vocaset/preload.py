import os
import re
import math
import saber
import torch
import pickle
import librosa
import numpy as np
from shutil import copyfile
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from scipy.io import wavfile
from copy import deepcopy
from saber import audio as audio_utils
from saber import stream as stream_utils
from saber.data.mesh.io import read_ply
from saber.utils.bilateral import BilateralFilter1D
from scipy.ndimage.filters import gaussian_filter1d
from collections import defaultdict
from speech_anime.tools import data_info
from .config import speaker_alias_dict
from . import mask
from ... import viewer
import deformation


_train_speakers = ["m0", "f0", "m1", "m2", "f1", "m3", "f2", "f3"]
_valid_speakers = ["m4", "f4"]
_test_speakers  = ["m5", "f5"]
_frame_id_re = re.compile(r"^sentence\d\d\.(\d\d\d\d\d\d)\.ply$")

# my splition
_single_speaker_valid_sents = [17, 18]

# metdadata
_metadata = [
    "speaker:str",
    "emotion:str",
    "sentence_id:int",
    "start_ts:float",
    "anime_minfi:int",
    "anime_maxfi:int",
    "anime_mints:float",
    "anime_maxts:float",
    "audio_samples:int",
    "npy_data_path:path",
    "sentence:str"
]

# speaker specific
_speaker_trim_dict = dict(
    m0={
        26: 8000,
        31: 5900,
        39: 5500,
    },
    m1={
        3:  12000,
        8:  8000,
        17: 7800,
        18: 10500,
        24: 8000,
        27: 10000,
        29: 10300,
        30: 10500,
        36: 12500,
        37: 12800,
        38: 13500,
    },
    m2={
        18: 8000,
        30: 7000,
        36: 8200,
        37: 10000,
        38: 5000,
    },
    m3={
        35: 4700,
        36: 9500,
        37: 3000,
    },
    m4={
        25: 16000,
        28: 10000,
        29: 0,
        30: 8000,
        35: 12500,
        36: 13000,
        37: 12500,
        38: 14000,
    },
    f0={
        17: 12000,
        19: 10000,
        35: 10000,
        36: 9800,
        38: 15000,
    },
    f1={
        17: 8700,
        18: 10000,
        19: 11000,
        24: 16410,
        26: 15000,
        28: 21500,
        38: 13500,
    },
    f2={
        17: 10000,
        19: 11000,
        28: 12000,
        35: 9900,
    },
    f3={
        0:  11500,
        9:  0,
        20: 10500,
        22: 8500,
        35: 10000,
        39: 8500,
    },
    f4={
        6:  11000,
        16: 12500,
        17: 8500,
        18: 7000,
        19: 9000,
        27: 5200,
        33: 7400,
        35: 5400,
        37: 8900,
        38: 12500,
        39: 8100,
    }
)

_must_silent_dict = dict(
    m3={
        37: 3000,
    }
)


with open(os.path.join(os.path.dirname(__file__), "mask", "voca_lower_face.txt")) as fp:
    line = fp.readline().strip().split()
    _lower_vert_ids = np.asarray([int(x) for x in line], np.int32)
    _other_vert_ids = np.asarray([
        x for x in range(5023) if x not in _lower_vert_ids
    ], np.int32)

# _filter_lower = BilateralFilter1D(factor=-0.5, distance_sigma=1.0, range_sigma=1.0, radius=5)
# _filter_other = BilateralFilter1D(factor=-0.5, distance_sigma=5.0, range_sigma=2.0, radius=10)

# load speaker templates
_speaker_template = dict()
for spk, subdir in speaker_alias_dict.items():
    _template_path = os.path.join(os.path.dirname(__file__), "templates", f"{subdir}.ply")
    _template = np.reshape(saber.mesh.io.read_ply(_template_path)[0], (-1, 3))
    # print(spk, _template_path)
    _speaker_template[spk] = {
        "template": np.copy(_template),
        "non_lower_face": np.copy(_template[_other_vert_ids])
    }


def _process_data(
    i_data,
    speaker, sent_id,
    output_prefix,
    path_wav,
    text,
    path_debug,
    sample_rate,
    target_db,
    silence_sec=0,
):
    output_wav = output_prefix + ".wav"
    output_vad = output_prefix + ".vad"
    output_txt = output_prefix + ".txt"
    # guard
    if (
        os.path.exists(output_wav) and
        os.path.exists(output_txt) and
        os.path.exists(output_vad)
    ):
        return

    # roughly process the signal
    signal = saber.audio.load(path_wav, sample_rate)
    denoised = saber.audio.denoise(signal, sample_rate, specific_key=str(i_data))

    # - pad or trim silence
    _manual_trim_dict = _speaker_trim_dict.get(speaker, dict())
    manual_trim = _manual_trim_dict.get(int(sent_id), 0)
    signal = signal[manual_trim:]
    denoised = denoised[manual_trim:]

    vad = saber.audio.vad.detect_speech(denoised, sample_rate, vad_mode=3)
    vad_signal = signal[vad > 0]
    if len(vad_signal) == 0:
        return os.path.basename(output_prefix)

    # rms normalize
    db = 20 * np.log10(np.sqrt(np.mean(vad_signal ** 2))+1e-10)
    max_db = 20 * np.log10(np.sqrt(max(vad_signal ** 2))+1e-10)
    delta_db = target_db - db
    if max_db + delta_db > 0:
        delta_db = -max_db
    scale_rms = np.power(10.0, delta_db / 20.0)
    signal = signal * scale_rms

    # - get vad tuples
    vad_pairs = saber.audio.vad.to_pairs(vad)
    # audio
    wavfile.write(output_wav, sample_rate, (signal * 32767.0).astype(np.int16))
    # text
    with open(output_txt, "w") as fp:
        fp.write(f"{text}\n")
    # vad
    with open(output_vad, "w") as fp:
        for pair in vad_pairs:
            fp.write(f"{pair[0]} {pair[1]}\n")
    # image
    saber.visualizer.plot(
        saber.visualizer.plot_item(signal, text),
        saber.visualizer.plot_item(vad, "vad"),
        saber.visualizer.plot_item(vad_signal, "vad"),
        aspect=5, file_path=path_debug+".png"
    )
    copyfile(output_wav, path_debug+".wav")
    return None


def clean_voca(root, clean_root, debug_root, sample_rate, target_db):
    i_data = 0
    futures = []
    executor = ProcessPoolExecutor(max_workers=8)
    missing = set()
    for i_spk, (spk, name) in enumerate(speaker_alias_dict.items()):
        _path_txt = os.path.join(root, "sentencestext", f"{name}.txt")
        _sentences = []
        with open(_path_txt) as fp:
            for line in fp:
                line = line.strip()
                if len(line) > 0:
                    _sentences.append(line)
        for i in range(1, 41):
            _txt = _sentences[i-1]
            _path_wav = os.path.join(root, "audio", name, f"sentence{i:02d}.wav")
            output_prefix = os.path.join(clean_root, spk, f"{spk}_{i:03d}")
            path_debug = os.path.join(debug_root, f"{spk}_{name}", f"sentence{i:03d}")
            os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
            os.makedirs(os.path.dirname(path_debug), exist_ok=True)
            if os.path.exists(_path_wav):
                # _process_data(
                #     i_data, spk, i-1, output_prefix,
                #     _path_wav, _txt, path_debug,
                #     sample_rate, target_db,
                # )
                futures.append(executor.submit(partial(
                    _process_data,
                    i_data, spk, i-1,
                    output_prefix,
                    _path_wav, _txt, path_debug,
                    sample_rate, target_db,
                )))
                i_data += 1
            else:
                missing.add(name)

    # process all tasks
    err_list = []
    progress = saber.log.tqdm(futures, desc='clean voca')
    for future in progress:
        err_name = future.result()
        if err_name is not None:
            err_list.append(err_name)
    with open(os.path.join(clean_root, "err_list.txt"), "w") as fp:
        for err in err_list:
            fp.write(f"{err}\n")


def preload_voca(
    root,
    clean_root,
    output_root,
    sample_rate,
    debug_audio,
    debug_video,
):

    all_info_dicts = []
    for spk, alias in speaker_alias_dict.items():
        saber.log.info("processing", spk)
        spk_root = os.path.join(root, "unposedcleaneddata", alias)
        if debug_video:
            viewer.set_template_mesh(os.path.join(root, "templates", f"{alias}.ply"))
        for si in saber.log.tqdm(range(1, 41), desc="sentence"):
            # if si != 2: continue

            if spk == "m5" and si == 26:  # data error! missing frame 1
                continue

            path = os.path.join(spk_root, "sentence{}".format(str(si).zfill(2)))
            if not os.path.exists(path):
                continue

            outp = data_info.get_path(output_root, spk, "neutral", si-1)
            imgp = os.path.join(output_root, "image", spk, f"{si-1:03d}.png")
            vidp = os.path.join(output_root, "video", spk, f"{si-1:03d}.mp4")
            info_dict = _collect(
                speaker     = spk,
                sent_id     = si - 1,
                path        = path,
                path_txt    = os.path.join(clean_root, spk, f"{spk}_{si:03d}.txt"),
                path_wav    = os.path.join(clean_root, spk, f"{spk}_{si:03d}.wav"),
                path_vad    = os.path.join(clean_root, spk, f"{spk}_{si:03d}.vad"),
                output_path = outp,
                image_path  = imgp,
                video_path  = vidp,
                # global settings
                sr          = sample_rate,
                debug_audio = debug_audio,
                debug_video = debug_video,
            )

            all_info_dicts.append(info_dict)

    os.makedirs(output_root, exist_ok=True)
    # if len(speakers) == 1:
    if False:
        # single speaker, split with sentences
        trainset = []
        validset = []
        for d in all_info_dicts:
            if d["sentence_id:int"] in _single_speaker_valid_sents:
                validset.append(deepcopy(d))
            else:
                trainset.append(deepcopy(d))
        saber.csv.write_csv(_metadata, trainset, os.path.join(output_root, "train.csv"))
        saber.csv.write_csv(_metadata, validset, os.path.join(output_root, "valid.csv"))
    else:
        # full speakers
        # 8 speaker for training, 2 speaker (last 20) for val, 2 speaker (last 20) for test
        trainset = []
        validset = []
        testset = []
        for d in all_info_dicts:
            if d["speaker:str"] in _train_speakers:
                trainset.append(deepcopy(d))
            elif d["speaker:str"] in _valid_speakers:
                if d["sentence_id:int"] >= 20:
                    validset.append(deepcopy(d))
                # else:
                #     trainset.append(deepcopy(d))
            elif d["speaker:str"] in _test_speakers:
                if d["sentence_id:int"] >= 20:
                    testset.append(deepcopy(d))
            else:
                raise ValueError("unknown speaker: {}".format(d["speaker:str"]))
        # save to csv
        saber.csv.write_csv(_metadata, trainset, os.path.join(output_root, "train.csv"))
        saber.csv.write_csv(_metadata, validset, os.path.join(output_root, "valid.csv"))
        saber.csv.write_csv(_metadata, testset,  os.path.join(output_root, "test.csv"))


def _read_sentences(txt_path):
    assert os.path.exists(txt_path), "failed to find: '{}'".format(txt_path)
    with open(txt_path) as fp:
        sents = []
        for line in fp:
            line = line.strip()
            if len(line) > 0:
                sents.append(line)
        return sents


def _interpolate(p0, p1, v0, v1, p, mode="linear"):
    if mode == "linear":
        a = (p - p0) / (p1 - p0)
        return v0 * (1.0 - a) + v1 * a
    else:
        raise NotImplementedError(f"'{mode}' mode is not supported!")


def _process_phonemes(phoneme_tuples, signal, sr):
    # align phonemes to feature
    last = 0
    phonemes = []
    for tup in phoneme_tuples:
        def append(cur, start, end, last):
            if start < last:
                saber.log.warn("a overlap bewteen phonemes!")
            if start > last:
                saber.log.warn("a gap between phonemes!")
                # phonemes.append([last, "#"])
            phonemes.append([end, cur])
            return end

        cur = tup[0]
        cur_start = tup[1] * 1000.0
        cur_end = tup[2] * 1000.0
        last = append(cur, cur_start, cur_end, last)

    if phonemes[-1][1] != "#":
        phonemes.append([cur_end, "#"])

    # generate vad according to phonemes
    idx = 0
    vad = np.zeros((len(signal)), dtype=np.float32)
    for i in range(len(vad)):
        ms = 1000.0 * float(i) / float(sr)
        while idx + 1 < len(phonemes) and ms > phonemes[idx][0]:
            idx += 1
        if phonemes[idx][1] != "#":
            vad[i] = 1.0
    return phonemes, vad


temp_verts, temp_faces = saber.mesh.read_mesh("speech_anime/datasets/vocaset/template/FLAME_sample.ply")


def _collect(
    speaker     ,
    sent_id     ,
    path        ,
    path_txt    ,
    path_wav    ,
    path_vad    ,
    output_path ,
    image_path  ,
    video_path  ,
    sr          ,
    debug_audio ,
    debug_video ,
):

    # get template and fitted zero exp
    anime_ts_delta = 100
    anime_ends_extra = 50
    anime_smooth_threshold = 150
    audio_silence_samples = sr // 2

    with open(path_txt) as fp:
        sent_txt = fp.readline().strip()

    if not os.path.exists(output_path + "_audio"):
        # ------------- #
        # process audio #
        # ------------- #
        start_ts = 0.0
        signal = saber.audio.load(path_wav, sr=sr)
        denoised = audio_utils.denoise(signal, sr)

        # must silent
        _silent_dict = _must_silent_dict.get(speaker, dict())
        _must_silent = _silent_dict.get(int(sent_id), 0)
        signal[:_must_silent] = 0
        denoised[:_must_silent] = 0

        # get vad
        vad_pairs = []
        with open(path_vad) as fp:
            for line in fp:
                line = line.strip()
                if len(line) > 0:
                    x, y = line.split()
                    vad_pairs.append((int(x), int(y)))
        vad = saber.audio.vad.from_pairs(vad_pairs, len(signal))

        # pad trimmed silence
        _manual_trim_dict = _speaker_trim_dict.get(speaker, dict())
        manual_trim = _manual_trim_dict.get(int(sent_id), 0)
        manual_trim_ms = 1000.0 * float(manual_trim) / float(sr)
        if manual_trim > 0:
            vad = np.pad(vad, [[manual_trim, 0]], "constant")
            signal = np.pad(signal, [[manual_trim, 0]], "constant")
            denoised = np.pad(denoised, [[manual_trim, 0]], "constant")
        denoised[vad == 0] = 0

        # pad silence
        pad = [0, 0]
        stt_smp = np.first_nonzero(vad, 0)
        end_smp = np.last_nonzero(vad, 0)
        if audio_silence_samples > stt_smp:
            pad[0] = audio_silence_samples - stt_smp
        if audio_silence_samples > len(signal) - end_smp:
            pad[1] = audio_silence_samples - len(signal) + end_smp
        vad          = np.pad(vad,          [pad], "constant")
        denoised     = np.pad(denoised,     [pad], "constant")
        signal = np.pad(signal, [pad], "constant")

        # trim silence
        stt_smp = np.first_nonzero(vad, 0)
        end_smp = np.last_nonzero(vad, 0)
        stt_smp = max(stt_smp - audio_silence_samples, 0)
        end_smp = min(end_smp + audio_silence_samples, len(signal))
        vad = vad[stt_smp:end_smp]
        denoised = denoised[stt_smp:end_smp]
        signal = signal[stt_smp:end_smp]

        # start ts and adjust
        start_ts = float(stt_smp * 1000.0) / float(sr) - float(pad[0] * 1000.0) / float(sr)

        # get the anime start, end tts
        anime_stt_ts = float(np.first_nonzero(vad, 0)) * 1000.0 / float(sr) + start_ts - anime_ts_delta - anime_ends_extra
        anime_end_ts = float(np.last_nonzero(vad, 0))  * 1000.0 / float(sr) + start_ts - anime_ts_delta + anime_ends_extra + 20
        anime_stt_fi = math.ceil(anime_stt_ts * 60.0 / 1000.0)
        anime_end_fi = math.floor(anime_end_ts * 60.0 / 1000.0)
        anime_stt_ts = anime_stt_fi * 1000.0 / 60.0  # update
        anime_end_ts = anime_end_fi * 1000.0 / 60.0  # update

        # ------------- #
        # process anime #
        # ------------- #
        # load ply frames
        os.makedirs(output_path, exist_ok=True)
        ply_files = saber.filesystem.find_files(path, r"sentence\d\d\.\d\d\d\d\d\d\.ply")
        # check all exist
        need_load_ply = False
        for fi, ply_file in enumerate(ply_files):
            save_path = os.path.join(output_path, "{}.npy".format(str(fi).zfill(6)))
            if not os.path.exists(save_path):
                need_load_ply = True
                break

        # need to read and process
        verts_seq = []
        if need_load_ply:
            # get template and non-lower-face verts
            assert speaker in _speaker_template
            spk_template = np.copy(_speaker_template[speaker]["template"])

            for fi, ply_file in enumerate(ply_files):
                frame_id = int(_frame_id_re.match(os.path.basename(ply_file)).group(1))
                assert frame_id == fi + 1, "fi {}, file {}".format(fi, ply_file)

                verts, _ = read_ply(ply_file)
                verts = np.reshape(verts, (-1, 3))
                verts_seq.append(verts)

            verts_seq = np.asarray(verts_seq)

            # small modification on template according to each sentence
            spk_template[_other_vert_ids] = verts_seq[:, _other_vert_ids].mean(axis=0)

            # remove template and get offsets
            verts_seq -= spk_template

            # pad frames
            anime_minfi = min(0, int(start_ts * 60.0 / 1000.0))
            anime_maxfi = max(len(verts_seq) - 1, int(len(signal) * 60.0 / float(sr)))

            def _clip_idx(fi):
                return min(max(fi, 0), len(verts_seq) - 1)

            # set template at silence
            th = anime_smooth_threshold
            for fi in range(anime_minfi, anime_maxfi + 1):
                ts = float(fi) * 1000.0 / 60.0
                if anime_stt_ts <= ts <= anime_end_ts:
                    to_save = verts_seq[_clip_idx(fi)]
                elif (ts <= anime_stt_ts - th) or (ts >= anime_end_ts + th):
                    to_save = np.zeros_like(spk_template)
                elif anime_stt_ts - th < ts < anime_stt_ts:
                    # blend from template to first frame
                    lower_p = anime_stt_ts - th
                    upper_p = anime_stt_ts
                    lower_v = np.zeros_like(spk_template)
                    upper_v = verts_seq[_clip_idx(anime_stt_fi)]
                    to_save = _interpolate(lower_p, upper_p, lower_v, upper_v, ts)
                elif anime_end_ts < ts < anime_end_ts + th:
                    # blend from last frame to template
                    lower_p = anime_end_ts
                    upper_p = anime_end_ts + th
                    lower_v = verts_seq[_clip_idx(anime_end_fi)]
                    upper_v = np.zeros_like(spk_template)
                    to_save = _interpolate(lower_p, upper_p, lower_v, upper_v, ts)
                else:
                    raise NotImplementedError("impossible!")

                save_path = os.path.join(output_path, "{}.npy".format(str(fi).zfill(6)))
                np.save(save_path, to_save.flatten())

            # filter
            # verts_seq[:, _lower_vert_ids] = _filter_lower(verts_seq[:, _lower_vert_ids], verbose=True)
            # verts_seq[:, _other_vert_ids] = _filter_other(verts_seq[:, _other_vert_ids], verbose=True)
            # verts_seq[:, _other_vert_ids] = gaussian_filter1d(
            #     input=verts_seq[:, _other_vert_ids],
            #     sigma=25,
            #     axis=0,
            # )

        # --------- #
        # dump data #
        # --------- #
        sr8k_signal = librosa.resample(signal, sr, 8000)
        sr8k_audio_denoised = librosa.resample(signal, sr, 8000)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path + "_audio", "wb") as fp:
            pickle.dump(dict(
                sr=sr,
                start_ts=start_ts,
                audio=signal,
                audio_denoised=denoised,
                audio_8k=sr8k_signal,
                audio_denoised_8k=sr8k_audio_denoised,
                # phonemes=phonemes,
            ), fp)

        # ---------- #
        # debug data #
        # ---------- #
        if debug_audio:
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            saber.visualizer.plot(
                saber.visualizer.plot_item(signal, title="original", sr=sr),  #, aligned_transcription=aligned_transcription),
                saber.visualizer.plot_item(denoised, title="denoised", sr=sr),  #, aligned_transcription=aligned_transcription),
                saber.visualizer.plot_item(vad, title="vad"),
                saber.visualizer.plot_item(
                    audio_utils.features.get("mel", signal, sr, win_size=1024, hop_size=128),
                ),
                saber.visualizer.plot_item(
                    audio_utils.features.get("mel", denoised, sr, win_size=1024, hop_size=128),
                ),
                file_path=image_path, aspect=6
            )
            audio_utils.save(os.path.splitext(image_path)[0] + '.wav', denoised, sr)

        # --------- #
        # visualize #
        # --------- #
        if debug_video and need_load_ply:
            def get_anime(shifted_l, shifted_r):
                ts = ((shifted_l+shifted_r)/2.0 * 1000.0) / float(sr)
                ts = ts - anime_ts_delta + start_ts
                pos = ts * 60.0 / 1000.0
                pos_lower = int(math.floor(pos))
                pos_upper = pos_lower + 1
                if pos_lower < anime_minfi:
                    pos_lower = anime_minfi
                    pos_upper = anime_minfi
                elif pos_upper > anime_maxfi:
                    pos_lower = anime_maxfi
                    pos_upper = anime_maxfi
                path_lower = os.path.join(output_path, "{}.npy".format(str(pos_lower).zfill(6)))
                path_upper = os.path.join(output_path, "{}.npy".format(str(pos_upper).zfill(6)))
                assert os.path.exists(path_lower), "invalid frame pos '{}'".format(pos_lower)
                assert os.path.exists(path_upper), "invalid frame pos '{}'".format(pos_upper)
                lower = np.load(path_lower)
                upper = np.load(path_upper)
                # bilinear
                a = float(pos - pos_lower)
                feat = lower * (1.0 - a) + upper * a
                return feat

            with open(output_path + "_audio", "rb") as fp:
                _vis_data = pickle.load(fp)
            _vis_images = []
            _vis_tslist = []
            _vis_frames = []
            sliding_size = 128 * 64 + (1024 - 128)
            fi = 0
            while int(fi * sr / 60.0) < len(_vis_data["audio"]):
                s = int(fi * sr / 60.0)
                e = s + sliding_size
                fi += 1
                _vis_tslist.append((s+e)/2.0 * 1000.0 / float(sr))
                _vis_frames.append(get_anime(s, e))
                wav = _vis_data["audio"][max(s, 0): min(e, len(_vis_data["audio"]))]
                if len(wav) == 0:
                    wav = np.zeros((sliding_size), np.float32)
                elif s < 0:
                    wav = np.pad(wav, [[-s, 0]], "constant")
                elif e > len(_vis_data["audio"]):
                    wav = np.pad(wav, [[0, e - len(_vis_data["audio"])]], "constant")
                mel = saber.audio.features.get("mel", wav, sr=sr, win_size=1024, hop_size=128, normalize=True)
                img = saber.visualizer.plot(saber.visualizer.plot_item(mel, vmin=0, vmax=1), aspect=1.0)
                _vis_images.append(img)

            _vis_true_data = dict(
                title         = "offsets + template",
                audio         = _vis_data["audio"],
                verts_off_3d  = _vis_frames,
                tslist        = _vis_tslist,
            )
            _imgdata = dict(
                title       = 'image',
                images      = _vis_images,
                tslist      = _vis_tslist,
            )
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            viewer.render_video([_vis_true_data, _imgdata], 60, sr, save_video=True, video_path=video_path, verbose=False)

    else:

        with open(output_path + "_audio", "rb") as fp:
            cached = pickle.load(fp)
            start_ts = cached["start_ts"]
            signal = cached["audio"]

        # if "audio_8k" not in cached or "audio_denoised_8k" not in cached or "audio_ps" not in cached:
        #     sr8k_signal = librosa.resample(cached["audio"], cached["sr"], 8000)
        #     sr8k_audio_denoised = librosa.resample(cached["audio_denoised"], cached["sr"], 8000)

        #     audio_sr_u4 = librosa.effects.pitch_shift(cached["audio"], cached["sr"], 4)
        #     audio_sr_u2 = librosa.effects.pitch_shift(cached["audio"], cached["sr"], 2)
        #     audio_sr_d2 = librosa.effects.pitch_shift(cached["audio"], cached["sr"], -2)
        #     audio_sr_d4 = librosa.effects.pitch_shift(cached["audio"], cached["sr"], -4)

        #     audio_8k_u4 = librosa.effects.pitch_shift(sr8k_signal, 8000, 4)
        #     audio_8k_u2 = librosa.effects.pitch_shift(sr8k_signal, 8000, 2)
        #     audio_8k_d2 = librosa.effects.pitch_shift(sr8k_signal, 8000, -2)
        #     audio_8k_d4 = librosa.effects.pitch_shift(sr8k_signal, 8000, -4)

        #     with open(output_path + "_audio", "wb") as fp:
        #         pickle.dump(dict(
        #             sr=cached["sr"],
        #             audio=cached["audio"],
        #             audio_8k=sr8k_signal,
        #             audio_denoised=cached["audio_denoised"],
        #             audio_denoised_8k=sr8k_audio_denoised,
        #             audio_ps_u4=audio_sr_u4,
        #             audio_ps_u2=audio_sr_u2,
        #             audio_ps_d2=audio_sr_d2,
        #             audio_ps_d4=audio_sr_d4,
        #             audio_8k_ps_u4=audio_8k_u4,
        #             audio_8k_ps_u2=audio_8k_u2,
        #             audio_8k_ps_d2=audio_8k_d2,
        #             audio_8k_ps_d4=audio_8k_d4,
        #             start_ts=cached["start_ts"],  # untouched
        #         ), fp)

    npy_files = saber.filesystem.find_files(output_path, r"-*\d+\.npy")
    frames = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in npy_files])
    anime_minfi = frames[0]
    anime_maxfi = frames[-1]

    for npy_file in npy_files:
        offsets = np.reshape(np.load(npy_file), (-1, 3))
        verts = offsets + temp_verts
        dist = verts[3531][1] - verts[3509][1]
        save_path = os.path.splitext(npy_file)[0] + "_lips_dist.npy"
        np.save(save_path, dist)

    # information
    info_dict = {
        "speaker:str": speaker,
        "emotion:str": "neutral",
        "sentence_id:int": sent_id,
        "start_ts:float": start_ts,
        "anime_minfi:int": anime_minfi,
        "anime_maxfi:int": anime_maxfi,
        "anime_mints:float": (anime_minfi * 1000.0 / 60.0),
        "anime_maxts:float": (anime_maxfi * 1000.0 / 60.0),
        "audio_samples:int": len(signal),
        "npy_data_path:path": output_path,
        "sentence:str": f"\"{sent_txt}\""
    }
    for k in _metadata:  # help check
        assert k in info_dict, "'{}' is missing!".format(k)
    return info_dict


def generate_dgrad(offsets_root, dgrad_root):
    speakers = _train_speakers + _valid_speakers + _test_speakers

    def _get_deform_grad(offsets, save_path, spk_template):
        offsets = np.reshape(offsets, (-1, 3))
        verts = spk_template + offsets
        dg = deformation.get_deform_grad(
            verts_a=spk_template,
            verts_b=verts,
            faces=spk_faces,
            eps=1e-6,
        )
        dg = np.reshape(dg, (-1, 9))
        dg[mask.non_face.non_face_tris] = 0
        np.save(save_path, dg.flatten(order='C').astype(np.float32))

    voca_root = os.path.dirname(__file__)
    for speaker in speakers:
        print("->", speaker)
        spk_template, spk_faces = saber.mesh.read_mesh(
            os.path.join(voca_root, "templates/{}.ply".format(speaker_alias_dict.get(speaker))),
            dtype=np.float32
        )

        for sent_id in range(1, 41):
            si = sent_id - 1
            src_root = os.path.join(offsets_root, "data", speaker, "neutral", f"{si:03d}")
            tar_root = os.path.join(dgrad_root, "data", speaker, "neutral", f"{si:03d}")

            # skip
            _src_audio = src_root + "_audio"
            _tar_audio = tar_root + "_audio"
            if not os.path.exists(_src_audio):
                continue

            if os.path.exists(_tar_audio):
                continue

            npy_files = []
            for _, _, files in os.walk(src_root):
                for f in files:
                    if re.match(r"^-*\d+.npy$", f):
                        npy_files.append(f)
                break
            npy_files = sorted(npy_files, key=lambda x: int(os.path.splitext(x)[0]))

            os.makedirs(tar_root, exist_ok=True)

            frames = []
            save_paths = []
            for f in saber.log.tqdm(npy_files):
                frames.append(np.load(os.path.join(src_root, f)))
                save_paths.append(os.path.join(tar_root, f))

            frames = gaussian_filter1d(frames, sigma=1, axis=0)

            for offset, save_path in zip(frames, save_paths):
                _get_deform_grad(offset, save_path, spk_template)

            lips_dist_files = saber.filesystem.find_files(src_root, r".*_lips_dist.npy", False, True)
            for lips_dist_file in lips_dist_files:
                save_path = os.path.join(tar_root, os.path.basename(lips_dist_file))
                copyfile(src=lips_dist_file, dst=save_path)

            copyfile(src=_src_audio, dst=_tar_audio)

    # copy
    copyfile(os.path.join(offsets_root, "train.csv"),
             os.path.join(dgrad_root,   "train.csv"))
    copyfile(os.path.join(offsets_root, "valid.csv"),
             os.path.join(dgrad_root,   "valid.csv"))


def pca_offsets(offsets_root, step=1):
    from psutil import virtual_memory
    from sklearn.decomposition import PCA, IncrementalPCA

    _filepath_compT = os.path.join(offsets_root, "pca", "compT.npy")
    _filepath_means = os.path.join(offsets_root, "pca", "means.npy")
    if os.path.exists(_filepath_compT) and os.path.exists(_filepath_means):
        saber.log.info("PCA of offsets if already calculated.")
        return

    # check memory
    mem = virtual_memory()
    available_gb = mem.available / (1024 * 1024 * 1024)
    if available_gb < 20:
        raise MemoryError(
            "To compuate pca of offsets, at least 20GB memory is necessary,"
            f" but only {available_gb:.1f} avaliable.\n"
            f"{' ' * 13}You can download the pre-trained pca from url in README.md"
        )

    csv_file = os.path.join(offsets_root, "train.csv")
    meta_data, info_list = saber.csv.read_csv(csv_file)

    npy_path_list = []
    for data_dict in saber.log.tqdm(info_list):
        data_dir = data_dict["npy_data_path:path"]
        for i, npy_path in enumerate(saber.filesystem.find_files(data_dir, r"-*\d+\.npy")):
            if step > 1 and i % step != 0:
                continue
            npy_path_list.append(npy_path)

    full_shape = (len(npy_path_list), len(np.load(npy_path_list[0])))
    all_verts = np.zeros(full_shape, dtype=np.float32)
    for r, npy_path in enumerate(saber.log.tqdm(npy_path_list, desc="pca, find frames")):
        all_verts[r] = np.load(npy_path)

    # pca
    saber.log.info("pca verts offsets...", all_verts.shape)
    pca = PCA(n_components=0.97, copy=False)
    with saber.log.timeit("pca fitting"):
        pca.fit(all_verts)
    print(pca.explained_variance_ratio_.cumsum()[-1], len(pca.explained_variance_ratio_))
    saber.log.info("compT: {}, means: {}".format(pca.components_.T.shape, pca.mean_.shape))
    os.makedirs(os.path.join(offsets_root, "pca"), exist_ok=True)
    np.save(_filepath_compT, pca.components_.T)
    np.save(_filepath_means, pca.mean_)


def pca_dgrad(dgrad_root, step=1):
    from psutil import virtual_memory
    from sklearn.decomposition import PCA, IncrementalPCA

    _filepath_scale_compT = os.path.join(dgrad_root, "pca", "scale_compT.npy")
    _filepath_scale_means = os.path.join(dgrad_root, "pca", "scale_means.npy")
    _filepath_rotat_compT = os.path.join(dgrad_root, "pca", "rotat_compT.npy")
    _filepath_rotat_means = os.path.join(dgrad_root, "pca", "rotat_means.npy")
    if (
        os.path.exists(_filepath_scale_compT) and
        os.path.exists(_filepath_scale_means) and
        os.path.exists(_filepath_rotat_compT) and
        os.path.exists(_filepath_rotat_means)
    ):
        saber.log.info("PCA of dgrad if already calculated.")
        return

    # check memory
    mem = virtual_memory()
    available_gb = mem.available / (1024 * 1024 * 1024)
    if available_gb < 70:
        raise MemoryError(
            "To compuate pca of dgrads, at least 70GB memory is necessary,"
            f" but only {available_gb:.1f} avaliable\n"
            f"{' ' * 13}You can download the pre-trained pca from url in README.md"
        )

    csv_file = os.path.join(dgrad_root, "train.csv")
    meta_data, info_list = saber.csv.read_csv(csv_file)

    npy_path_list = []
    for data_dict in saber.log.tqdm(info_list):
        data_dir = data_dict["npy_data_path:path"]
        for i, npy_path in enumerate(saber.filesystem.find_files(data_dir, r"-*\d+\.npy")):
            if step > 1 and i % step != 0:
                continue
            npy_path_list.append(npy_path)

    all_scale = np.zeros((len(npy_path_list), 9976*6), dtype=np.float32)
    all_rotat = np.zeros((len(npy_path_list), 9976*3), dtype=np.float32)
    for r, npy_path in enumerate(saber.log.tqdm(npy_path_list, desc="pca, find frames")):
        dg = np.reshape(np.load(npy_path), (-1, 9))
        all_scale[r] = dg[:, :6].flatten()
        all_rotat[r] = dg[:, 6:].flatten()

    # pca
    # scale
    saber.log.info("pca scale...")
    pca = PCA(0.97, copy=False)
    pca.fit(all_scale)
    print('scale', pca.explained_variance_ratio_.cumsum()[-1])
    saber.log.info("compT: {}, means: {}".format(pca.components_.T.shape, pca.mean_.shape))
    os.makedirs(os.path.join(dgrad_root, "pca"), exist_ok=True)
    np.save(os.path.join(dgrad_root, "pca", "scale_compT.npy"), pca.components_.T)
    np.save(os.path.join(dgrad_root, "pca", "scale_means.npy"), pca.mean_)
    del pca

    # rotat
    saber.log.info("pca rotat...")
    pca = PCA(0.97, copy=False)
    pca.fit(all_rotat)
    print('rotat', pca.explained_variance_ratio_.cumsum()[-1])
    saber.log.info("compT: {}, means: {}".format(pca.components_.T.shape, pca.mean_.shape))
    os.makedirs(os.path.join(dgrad_root, "pca"), exist_ok=True)
    np.save(os.path.join(dgrad_root, "pca", "rotat_compT.npy"), pca.components_.T)
    np.save(os.path.join(dgrad_root, "pca", "rotat_means.npy"), pca.mean_)
    del pca
