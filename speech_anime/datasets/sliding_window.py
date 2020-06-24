import os
import math
import saber
import torch
import pickle
import librosa
import numpy as np
from saber import (
    audio as audio_utils,
    stream as stream_utils,
)
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from saberspeech.datasets.tools import get_features, data_info
from saberspeech.datasets.speech_anime import SpeechAnimeDataset


class DatasetSlidingWindow(SpeechAnimeDataset):

    def __init__(self, hparams: saber.ConfigDict, training: bool):
        super().__init__(hparams, training)
        # audio feature
        self._sr = hparams.audio.sample_rate
        self._feat_name = hparams.audio.feature.name
        self._win_size = hparams.audio[self._feat_name].win_size
        self._hop_size = hparams.audio[self._feat_name].hop_size
        self._feat_frames = hparams.audio.feature.sliding_window_frames
        self._sliding_size = self._hop_size*(self._feat_frames-1)+self._win_size
        assert isinstance(self._sr, int)
        assert isinstance(self._feat_frames, int)
        assert isinstance(self._win_size, float)
        assert isinstance(self._hop_size, float)
        # assert self._sr == self.info_list[0]["sample_rate:int"]

        # anime feature
        self._fps = hparams.anime.fps
        self._face_type = hparams.model.face_data_type
        self._pred_type = hparams.model.prediction_type

        # set audio random shifting
        self._time_shifting = 0.5 / self._fps

        # initilaize coordinates
        self.coordinates = []
        extra_samples = self._sr // 3
        delta_samples = float(self._sr) / float(self._fps)
        sliding_size = int(self._sr * self._sliding_size)
        for i, info in enumerate(self.info_list):
            stt_sp = 0 - extra_samples
            end_sp = info["audio_samples:int"] + extra_samples

            left = stt_sp
            while left + sliding_size <= end_sp:
                s = math.ceil(left)
                e = s + sliding_size
                self.coordinates.append(dict(
                    data_id = i,
                    range   = (s, e)
                ))
                left += delta_samples

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, i_frame):
        # get next frame
        j_frame = i_frame + 1
        # if reach end of one data
        if (
            j_frame == len(self.coordinates) or       # out of range
            (self.coordinates[i_frame]["data_id"] !=  # not same sentence
             self.coordinates[j_frame]["data_id"])
        ):
            j_frame = i_frame
            i_frame = j_frame - 1
        # info of adjcent data_i and data_j
        i_info   = self.coordinates[i_frame]
        j_info   = self.coordinates[j_frame]
        data_id  = i_info["data_id"]
        l0, r0   = i_info["range"]
        l1, r1   = j_info["range"]
        assert i_info["data_id"] == j_info["data_id"]
        # random shift audio
        audio_shifting = int(self._time_shifting * self._sr)
        shift   = np.random.randint(-audio_shifting, audio_shifting + 1)
        l0, r0  = l0 + shift, r0 + shift
        l1, r1  = l1 + shift, r1 + shift

        # data path, speaker id, emotion id
        spk_id = self.get_speaker_id(self.info_list[data_id]["speaker:str"])
        emo_id = self.get_emotion_id(self.info_list[data_id]["emotion:str"])

        start_ts = self.info_list[data_id]["start_ts:float"]
        anime_minfi = self.info_list[data_id]["anime_minfi:int"]
        anime_maxfi = self.info_list[data_id]["anime_maxfi:int"]

        # get data
        data_path = self.info_list[data_id]["npy_data_path:path"]
        with open(data_path + "_audio", "rb") as fp:
            data = pickle.load(fp)

        assert self._sr == data["sr"],\
            f"sample_rate is not same! hparams {self._sr}, data {data['sr']}"

        # signal
        sr = data["sr"]
        signal = data["audio"]
        # # make sure not to modify this
        # denoised = np.copy(data["audio_denoised"])

        audio_feat_args = dict(
            force_preemph=None,
            signal_noise=None,
            feat_extra=None,
            feat_scale=None,
            feat_noise=None,
            feat_tremolo=None,
            feat_dropout=None,
        )
        pitch_shifted = False

        # get feature on-fly
        if self.training:
            # choose signal
            rand_reverb = self.hparams.audio.feature.random_reverb
            rand_ps = self.hparams.audio.feature.random_pitch_shift
            rand_noise = self.hparams.audio.feature.random_noise
            rand_preemph = self.hparams.audio.feature.random_preemph

            source_list = ["audio", "audio_denoised", "audio_8k", "audio_denoised_8k"]
            if rand_reverb: source_list.append("audio_reverb")
            if rand_ps: source_list.extend(["audio_ps", "audio_8k_ps"])
            source_type = np.random.choice(source_list)

            if source_type.find("_8k") >= 0:
                sr = 8000

            # print(source_type)
            if source_type in ["audio", "audio_reverb", "audio_denoised", "audio_8k", "audio_denoised_8k"]:
                signal = data[source_type]
            elif source_type in ["audio_ps", "audio_8k_ps"]:
                pitch_shifted = True
                suffix_list = ["_u4", "_u2", "_d2", "_d4"]
                signal = data[source_type+np.random.choice(suffix_list)]
            else:
                raise ValueError("unknown source type: {}".format(source_type))

            # random noise
            if rand_noise is not None:
                assert rand_noise > 0
                noise_type = np.random.choice(["none", "white"])
                if noise_type == "white":
                    noise_scale = np.random.uniform(rand_noise / 5, rand_noise)
                    audio_feat_args["signal_noise"] = "{}@{}".format(noise_type, noise_scale)

            # random preemphasis
            if rand_preemph is not None and rand_preemph > 0:
                audio_feat_args["force_preemph"] = np.random.uniform(0, rand_preemph)

            # print(f"wav: src {source_type}, noise {noise_detail}, preemph {preemph}")

        # random augment for mel
        ex_time = 0
        if self.training:
            mel_augment = self.hparams.audio.feature.random_mel_extra
            if mel_augment is not None:
                ex_feat, ex_time = mel_augment
                ex_feat = np.random.randint(-abs(ex_feat), abs(ex_feat)+1)
                ex_time = np.random.randint(-abs(ex_time), abs(ex_time)+1)
                # pitch is already shifted
                if pitch_shifted:
                    ex_feat = 0
                audio_feat_args["feat_extra"] = (ex_feat, ex_time)
            # random scale for mel
            mel_scale = self.hparams.audio.feature.random_mel_scale
            if mel_scale is not None:
                assert 0 <= mel_scale <= 0.2
                len_feat = self.hparams.audio.mel.n_mels
                _scale = np.sin(np.linspace(0, np.pi * 2, num=len_feat)
                                * np.random.uniform(-np.pi/2, np.pi/2)
                                + np.random.uniform(0, np.pi)) * mel_scale
                _scale = np.expand_dims(np.exp(_scale), 1)
                audio_feat_args["feat_scale"] = _scale
            # random noise for mel
            mel_noise = self.hparams.audio.feature.random_mel_noise
            if mel_noise is not None:
                shape = [self.hparams.audio.mel.num_mels, self.feat_frames]
                mel_noise = np.random.normal(0.0, mel_noise, size=shape)
                audio_feat_args["feat_noise"] = mel_noise
            # random dropout for mel
            mel_dropout = self.hparams.audio.feature.random_mel_dropout
            if mel_dropout is not None:
                mel_dropout = np.random.uniform(0, mel_dropout)
            audio_feat_args["feat_dropout"] = mel_dropout
            # random tremolo
            mel_tremolo = self.hparams.audio.feature.get("random_mel_tremolo", None)
            if mel_tremolo is not None:
                if np.random.uniform() < 0.5:
                    mel_tremolo = np.random.uniform(0, mel_tremolo)
                else:
                    mel_tremolo = None
            audio_feat_args["feat_tremolo"] = mel_tremolo

        def get_anime(shifted_l, shifted_r):
            assert self._face_type == "dgrad_3d"
            ts_delta = self.hparams.anime.feature.ts_delta
            if self._face_type == "dgrad_3d":
                ts = self.sample_to_ms((shifted_l+shifted_r)/2)
                ts = ts - ts_delta + start_ts
                pos = ts * self.hparams.anime.fps / 1000.0
                pos_lower = int(math.floor(pos))
                pos_upper = pos_lower + 1
                if pos_lower < anime_minfi:
                    pos_lower = anime_minfi
                    pos_upper = anime_minfi
                elif pos_upper > anime_maxfi:
                    pos_lower = anime_maxfi
                    pos_upper = anime_maxfi
                path_lower = os.path.join(data_path, "{}.npy".format(str(pos_lower).zfill(6)))
                path_upper = os.path.join(data_path, "{}.npy".format(str(pos_upper).zfill(6)))
                assert os.path.exists(path_lower), "invalid frame pos '{}'".format(pos_lower)
                assert os.path.exists(path_upper), "invalid frame pos '{}'".format(pos_upper)
                lower = np.load(path_lower)
                upper = np.load(path_upper)
                # bilinear
                a = float(pos - pos_lower)
                feat = lower * (1.0 - a) + upper * a
                # weight
                weight = 1.0
                path_lower = os.path.join(data_path, "{}_lips_dist.npy".format(str(pos_lower).zfill(6)))
                path_upper = os.path.join(data_path, "{}_lips_dist.npy".format(str(pos_upper).zfill(6)))
                assert os.path.exists(path_lower), "invalid frame pos '{}'".format(pos_lower)
                assert os.path.exists(path_upper), "invalid frame pos '{}'".format(pos_upper)
                lower = np.load(path_lower)
                upper = np.load(path_upper)
                dist = lower * (1.0 - a) + upper * a
                weight = np.exp((0.002 - dist) * 50) * 2
            else:
                raise NotImplementedError()
            return feat.astype(np.float32), np.float32(weight)

        # get feature
        ph_aligned = data.get("phonemes")
        feat0, ph0, wav0, random_args = self._audio_features(
            signal, l0, r0, ph_aligned,
            sample_rate=sr,
            training=self.training, **audio_feat_args
        )
        # print(random_args)
        feat1, ph1, wav1, _ = self._audio_features(
            signal, l1, r1, ph_aligned,
            sample_rate=sr,
            training=self.training, **audio_feat_args,
            random_args=random_args  # use same random_args
        )

        ret = {
            "sr": sr,
            "emotion_id": emo_id,
            "speaker_id": spk_id,
            "signal_0": wav0,
            "signal_1": wav1,
            "frame_id_0": i_frame,
            "frame_id_1": j_frame,
            "audio_feat_0": feat0,
            "audio_feat_1": feat1,
        }

        anime_feat0, ret["anime_weight_0"] = get_anime(l0, r0)
        anime_feat1, ret["anime_weight_1"] = get_anime(l1, r1)
        anime_feat0 = np.reshape(anime_feat0, (-1, 9))
        anime_feat1 = np.reshape(anime_feat1, (-1, 9))
        ret["dgrad_3d_scale_0"] = np.expand_dims(anime_feat0[:, :6], axis=0)
        ret["dgrad_3d_rotat_0"] = np.expand_dims(anime_feat0[:, 6:], axis=0)
        ret["dgrad_3d_scale_1"] = np.expand_dims(anime_feat1[:, :6], axis=0)
        ret["dgrad_3d_rotat_1"] = np.expand_dims(anime_feat1[:, 6:], axis=0)

        return ret

    def collate(self, batch):
        # pad signal to same length
        max_samples = max([
            max(len(d["signal_0"]), len(d["signal_1"]))
            for d in batch
        ])
        # collate
        half = len(batch)
        full = half * 2
        real_batch = []
        for i in range(full):
            flag = int(i >= half)
            k = i if i < half else i - half
            wav = batch[k]["signal_{}".format(flag)]
            if len(wav) < max_samples:
                wav = np.pad(wav, [[0, max_samples-len(wav)]], "constant")
            tup = {
                "sr":            batch[k]["sr"],
                "signal":        wav,
                "speaker_id":    batch[k]["speaker_id"],
                "emotion_id":    batch[k]["emotion_id"],
            }
            for key in batch[k]:
                if key.split("_")[-1] == f"{flag}":
                    name = "_".join(key.split("_")[:-1])
                    if name not in tup:
                        tup[name] = batch[k][key]
            real_batch.append(tup)
        real_batch = self.default_collate(real_batch)
        return real_batch

    def information(self):
        return "{} speakers".format(self.num_speakers)

    @classmethod
    def frame_in_range(cls, frame_idx, sliding_size, start, end):
        return start + cls.frame_to_sample(frame_idx) + sliding_size <= end

    @classmethod
    def fetch_audio_features(cls, signal, hparams=None):
        if hparams is not None and cls.hparams is None:
            cls.hparams = hparams

        # check wav range
        assert -1.0 <= signal.min() and signal.max() <= 1.0

        # generate for overlapped frames
        feat_name = cls.hparams.audio.feature.name

        # generate all features for animation
        frames = cls.hparams.audio.feature.sliding_window_frames
        win_size = cls.hparams.audio[feat_name].win_size
        hop_size = cls.hparams.audio[feat_name].hop_size
        if isinstance(win_size, float):
            win_size = int(win_size * cls.hparams.audio.sample_rate)
        if isinstance(hop_size, float):
            hop_size = int(hop_size * cls.hparams.audio.sample_rate)
        sliding_size = hop_size*(frames-1)+win_size

        idx, ts_list, feat_list, eng_list = -1.0, [], [], []
        ts_delta = cls.hparams.anime.feature.ts_delta

        while cls.frame_in_range(idx, sliding_size, 0, len(signal) + sliding_size * 2):
            m = math.floor(cls.frame_to_sample(idx))
            e = m + sliding_size // 2
            s = e - sliding_size
            ts = cls.sample_to_ms((s+e)/2)
            ts -= ts_delta  # left shift ts_delta
            ts = int(round(ts))
            # get window signal
            part_wav = signal[max(0, s): min(len(signal), e)]
            if len(part_wav) == 0:
                part_wav = np.zeros((sliding_size), np.float32)
            elif s < 0:
                part_wav = np.pad(part_wav, [[-s, 0]], "constant")
            elif e > len(signal):
                part_wav = np.pad(part_wav, [[0, e-len(signal)]], "constant")
            assert len(part_wav) == sliding_size, f"signal length {len(part_wav)} != {sliding_size}."
            # get features
            energy = librosa.feature.rms(y=part_wav, frame_length=win_size, hop_length=hop_size, center=False)
            feat = cls._audio_features(part_wav)[0]
            # append
            ts_list.append(ts)
            eng_list.append(energy)
            feat_list.append(feat)
            idx += 1.0

        return dict(
            tslist = ts_list,
            energy = np.asarray(eng_list).astype(np.float32),
            audio_feat = np.asarray(feat_list).astype(np.float32),
        )

    @classmethod
    def _audio_features(
        cls, signal,
        stt=None, end=None,
        ph_aligned=None,
        force_preemph=None,
        signal_noise=None,
        feat_extra=None,
        feat_scale=None,
        feat_noise=None,
        feat_tremolo=None,
        feat_dropout=None,
        training=False,
        sample_rate=None,
        random_args=None,
    ):
        if not training:
            force_preemph = None
            signal_noise = None
            feat_extra = None
            feat_scale = None
            feat_noise = None
            feat_tremolo = None
            feat_dropout = None

        audio_config = deepcopy(cls.hparams.audio)
        feat_name = audio_config.feature.name
        if force_preemph is not None:
            audio_config[feat_name].set_key("preemphasis", force_preemph)

        # update sample rate
        sr = audio_config.sample_rate
        if sample_rate is None:
            sample_rate = sr
        if sample_rate != sr:
            assert isinstance(audio_config.mel.win_size, float)
            assert isinstance(audio_config.mel.hop_size, float)
            # update stt and end
            if stt and end is not None:
                length = int((end - stt) * sample_rate / sr)
                stt = int(stt * sample_rate / sr)
                end = stt + length
            else:
                if stt is not None: stt = int(stt * sample_rate / sr)
                if end is not None: end = int(end * sample_rate / sr)
            sr = sample_rate
            audio_config.set_key("sample_rate", sample_rate)

        if stt is None: stt = 0
        if end is None: end = len(signal)

        # set win_size or hop_size for each
        if isinstance(audio_config[feat_name].get("win_size"), float):
            audio_config[feat_name].set_key(
                "win_size",
                int(sr * audio_config[feat_name]["win_size"])
            )
        if isinstance(audio_config[feat_name].get("hop_size"), float):
            audio_config[feat_name].set_key(
                "hop_size",
                int(sr * audio_config[feat_name]["hop_size"])
            )

        feat, phs, wav, random_args = get_features.windowed_features(
            signal=signal,
            signal_stt=stt,
            signal_end=end,
            # for feature
            audio_config=audio_config,
            # framewise phonemes,
            ph_aligned=ph_aligned,
            # augment args
            signal_noise=signal_noise,
            feat_extra=feat_extra,
            feat_scale=feat_scale,
            feat_noise=feat_noise,
            feat_tremolo=feat_tremolo,
            feat_dropout=feat_dropout,
            random_args=random_args
        )

        phs = None

        feat = np.transpose(feat, (2, 1, 0))
        return feat, phs, wav, random_args
