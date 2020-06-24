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


class DatasetWin2Frm(SpeechAnimeDataset):
    trainset_scale_compT = None
    trainset_scale_means = None
    trainset_rotat_compT = None
    trainset_rotat_means = None

    def __init__(self, hparams: saber.ConfigDict, training: bool):
        super().__init__(hparams, training)
        # initialize hparams
        self.feat_names = deepcopy(hparams.audio.feature.feat_names)
        self.win_size = hparams.audio[self.feat_names[0].split("-")[0]].win_size
        self.hop_size = hparams.audio[self.feat_names[0].split("-")[0]].hop_size
        if isinstance(self.win_size, float):
            self.win_size = int(self.win_size * hparams.audio.sample_rate)
        if isinstance(self.hop_size, float):
            self.hop_size = int(self.hop_size * hparams.audio.sample_rate)
        self.from_disk = hparams.audio.feature.from_disk
        self.feat_frames = hparams.audio.feature.sliding_window
        self.sliding_size = self.hop_size*(self.feat_frames-1)+self.win_size
        self.blend_type = hparams.anime.feature.blend_type
        # check
        assert isinstance(self.feat_names, (tuple, list))
        # for name in self.feat_names:
        #     win_size = hparams.audio[name.split("-")[0]].win_size
        #     hop_size = hparams.audio[name.split("-")[0]].hop_size
        #     assert self.win_size == win_size, "feature's win_size not same!"
        #     assert self.hop_size == hop_size, "feature's hop_size not same!"

        # set audio random shifting
        self.audio_shifting = math.floor(hparams.audio.sample_rate/hparams.anime.fps/2)

        # # ignore some data!
        # self.info_list = [
        #     info for info in self.info_list
        #     if not (info["speaker:str"] == "m4" and info["sentence_id:int"] == 29)
        # ]

        # get all noise
        if training:
            self.sr16k_noises = [
                saber.audio.load(f, 16000)
                for f in saber.filesystem.find_files("saber/assets/noise/processed/sr16k", r".*.wav")
            ]
            self.sr8k_noises = [
                saber.audio.load(f, 8000)
                for f in saber.filesystem.find_files("saber/assets/noise/processed/sr8k", r".*.wav")
            ]
            print(f"sr16k noises: {len(self.sr16k_noises)}")
            print(f"sr8k noises: {len(self.sr8k_noises)}")

        # initilaize coordinates
        self.coordinates = []
        extra_samples = hparams.audio.sample_rate // 3
        delta_samples = float(hparams.audio.sample_rate) / float(hparams.anime.fps)
        for i, info in enumerate(self.info_list):
            stt_sp = 0 - extra_samples
            end_sp = info["audio_samples:int"] + extra_samples

            left = stt_sp
            while left + self.sliding_size <= end_sp:
                s = math.ceil(left)
                e = s + self.sliding_size
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
        shift   = np.random.randint(-self.audio_shifting, self.audio_shifting + 1)
        l0, r0  = l0 + shift, r0 + shift
        l1, r1  = l1 + shift, r1 + shift

        # data path, speaker id, emotion id
        spk_id = self.get_speaker_id(self.info_list[data_id]["speaker:str"])
        emo_id = self.get_emotion_id(self.info_list[data_id]["emotion:str"])
        if not self.training:
            # unseen speaker, set speaker id as 0
            spk_id = 0

        start_ts = self.info_list[data_id]["start_ts:float"]
        anime_minfi = self.info_list[data_id]["anime_minfi:int"]
        anime_maxfi = self.info_list[data_id]["anime_maxfi:int"]

        # get data
        data_path = self.info_list[data_id]["npy_data_path:path"]
        with open(data_path + "_audio", "rb") as fp:
            data = pickle.load(fp)

        assert self.hparams.audio.sample_rate == data["sr"],\
            "sample_rate is not same! hparams {}, data {}".format(
                self.hparams.audio.sample_rate, data["sr"]
        )

        # signal
        sr = data["sr"]
        signal = data["audio"]
        denoised = np.copy(data["audio_denoised"])  # make sure not to modify this
        stretch_rate = 1

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
        if self.training and (not self.from_disk):
            # choose signal
            rand_reverb = self.hparams.audio.feature.random_reverb
            rand_ps = self.hparams.audio.feature.random_pitch_shift
            rand_ts = self.hparams.audio.feature.random_time_stretch
            rand_noise = self.hparams.audio.feature.random_noise
            rand_preemph = self.hparams.audio.feature.random_preemph

            source_list = ["audio", "audio_denoised", "audio_8k", "audio_denoised_8k"]
            if rand_reverb: source_list.append("audio_reverb")
            if rand_ps: source_list.extend(["audio_ps", "audio_8k_ps"])
            if rand_ts: source_list.append("audio_ts")
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
            elif source_type == "audio_ts":
                suffix_list = [0.8, 1.2]
                stretch_rate = np.random.choice(suffix_list)
                signal = data["{}{}".format(source_type, stretch_rate)]
                # adjust l, r
                og_l0, og_r0 = l0, r0
                og_l1, og_r1 = l1, r1
                m0 = int((l0+r0)/2/stretch_rate)
                m1 = int((l1+r1)/2/stretch_rate)
                l0 = m0 - self.sliding_size // 2
                r0 = l0 + self.sliding_size
                l1 = m1 - self.sliding_size // 2
                r1 = l1 + self.sliding_size
                if not (0 <= l0 and r0 < len(signal) and 0 <= l1 and r1 < len(signal)):
                    # print("resume time stretch")
                    l0, r0 = og_l0, og_r0
                    l1, r1 = og_l1, og_r1
                    stretch_rate = 1
                    signal = data["audio"]
            else:
                raise ValueError("unknown source type: {}".format(source_type))

            # random noise
            if rand_noise is not None:
                assert rand_noise > 0
                noise_type = np.random.choice(["none", "white", "real"])
                if noise_type == "real":
                    if sr == 16000:
                        audio_feat_args["signal_noise"] = np.random.choice(self.sr16k_noises)
                    elif sr == 8000:
                        audio_feat_args["signal_noise"] = np.random.choice(self.sr8k_noises)
                    else:
                        raise NotImplementedError()
                else:
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

            # print(f"mel: scale {mel_scale}, noise {mel_noise}, drop {mel_dropout}, augment {mel_augment}")

        def get_anime(shifted_l, shifted_r):
            assert self.hparams.anime.feature.using_verts,\
                "voca only support 'using_verts=True'"
            ts_delta = self.hparams.anime.feature.ts_delta
            if self.hparams.anime.feature.using_verts:
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

        def get_anime_win(frames):
            _info = data_info.get_info(data_path)
            _path = os.path.join(self.pca_path_prefix, _info.speaker, _info.emotion, str(_info.sent_id).zfill(3) + ".npy")
            _full = np.load(_path)
            if len(_full) >= frames:
                _pos = np.random.randint(0, len(_full) - frames+1)
                return _full[_pos: _pos+frames].T.astype(np.float32)
            else:
                _pad = frames - len(_full)
                _full = np.pad(_full, [[_pad//2, _pad-_pad//2], [0, 0]], "constant")
                return _full.T.astype(np.float32)

        # def get_part_energy(left, right):
        #     return librosa.feature.rms(
        #         y=signal[left: right],
        #         frame_length=self.win_size,
        #         hop_length=self.hop_size,
        #         center=False
        #     )

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
            "frame_id_0": i_frame,
            "frame_id_1": j_frame,
            "signal_0": wav0,
            "signal_1": wav1,
            "energy_0": 0,
            "energy_1": 0,
            "audio_feat_0": feat0,
            "audio_feat_1": feat1,
        }

        symbol_type = self.hparams.text.type
        if symbol_type not in [None, "char"]:
            ret["phonemes_0"], ret["viseme_weight_0"] = ph0, 1.0
            ret["phonemes_1"], ret["viseme_weight_1"] = ph1, 1.0

        ret["anime_feat_0"], ret["anime_weight_0"] = get_anime(l0, r0)
        ret["anime_feat_1"], ret["anime_weight_1"] = get_anime(l1, r1)

        if self.hparams.anime.feature.anime_window > 0:
            raise NotImplementedError()
            anime_win = get_anime_win(self.hparams.anime.feature.anime_window)
            ret["anime_win_0"] = anime_win
            ret["anime_win_1"] = anime_win

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
                "energy":        batch[k]["energy_{}".format(flag)],
                "audio_feat":    batch[k]["audio_feat_{}".format(flag)],
                "anime_feat":    batch[k]["anime_feat_{}".format(flag)],
                "speaker_id":    batch[k]["speaker_id"],
                "semantic_id":   batch[k]["emotion_id"],
                "frame_id":      batch[k]["frame_id_{}".format(flag)],
                "anime_weight":  batch[k]["anime_weight_{}".format(flag)],
            }
            if "phonemes_0" in batch[k]:
                tup["phonemes"] = batch[k]["phonemes_{}".format(flag)]
                tup["viseme_weight"] = batch[k]["viseme_weight_{}".format(flag)]
            if "anime_win_0" in batch[k]:
                tup["anime_win"] = batch[k]["anime_win_{}".format(flag)]
            if "denoised_audio_feat_0" in batch[k]:
                tup["denoised_feat"] = batch[k]["denoised_audio_feat_{}".format(flag)]
            real_batch.append(tup)
        real_batch = self.default_collate(real_batch)
        return real_batch

    def information(self):
        return "{} speakers".format(self.num_speakers)

    @classmethod
    def frame_in_range(cls, frame_idx, sliding_size, start, end):
        return start + cls.frame_to_sample(frame_idx) + sliding_size <= end

    @classmethod
    def fetch_audio_features(cls, signal):
        # check wav range
        assert -1.0 <= signal.min() and signal.max() <= 1.0

        # generate for overlapped frames
        feat_names = cls.hparams.audio.feature.feat_names
        if not isinstance(feat_names, (tuple, list)):
            feat_names = [feat_names]

        # generate all features for animation
        frames = cls.hparams.audio.feature.sliding_window
        win_size = cls.hparams.audio[feat_names[0].split("-")[0]].win_size
        hop_size = cls.hparams.audio[feat_names[0].split("-")[0]].hop_size
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
        feat_names = audio_config.feature.feat_names
        if force_preemph is not None:
            for name in feat_names:
                audio_config[name].set_key("preemphasis", force_preemph)

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
        for name in audio_config.feature.feat_names:
            if isinstance(audio_config[name].get("win_size"), float):
                audio_config[name].set_key(
                    "win_size",
                    int(sr * audio_config[name]["win_size"])
                )
            if isinstance(audio_config[name].get("hop_size"), float):
                audio_config[name].set_key(
                    "hop_size",
                    int(sr * audio_config[name]["hop_size"])
                )
            # print(sr, audio_config[name]["win_size"])
            # print(sr, audio_config[name]["hop_size"])
            # quit()

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

        if phs is not None and cls.hparams.text.type not in [None, "char"]:
            phs = np.asarray([
                saber.text.PhonemeNopad.convert_stress_to_typeid(ph, cls.hparams.text.type)
                for ph in phs
            ], dtype=np.int64)
        else:
            phs = None

        return feat, phs, wav, random_args

    @property
    def pca_path_prefix(self):
        return os.path.join(self.root, "pca", "_".join(os.path.splitext(x)[0] for x in self.csv_files))

    def pca(self):
        scale_compT = np.load("experiments/pca/scale_compT.npy")
        scale_means = np.load("experiments/pca/scale_means.npy")
        rotat_compT = np.load("experiments/pca/rotat_compT.npy")
        rotat_means = np.load("experiments/pca/rotat_means.npy")
        return scale_compT, scale_means, rotat_compT, rotat_means


        # using cached
        cached_scale = self.pca_path_prefix + "_scale.pkg"
        cached_rotat = self.pca_path_prefix + "_rotat.pkg"
        if os.path.exists(cached_scale) and os.path.exists(cached_rotat):
            with open(cached_scale, "rb") as fp:
                cached = pickle.load(fp)
                scale_compT, scale_means = cached["compT"], cached["means"]
            with open(cached_rotat, "rb") as fp:
                cached = pickle.load(fp)
                rotat_compT, rotat_means = cached["compT"], cached["means"]
            saber.log.info("scale, compT: {}, means: {}".format(scale_compT.shape, scale_means.shape))
            saber.log.info("rotat, compT: {}, means: {}".format(rotat_compT.shape, rotat_means.shape))
        else:
            step = 1
            npy_path_list = []
            for data_dict in self.info_list:
                data_dir = data_dict["npy_data_path:path"]
                for i, npy_path in enumerate(saber.filesystem.find_files(data_dir, r".*\.npy")):
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
            from sklearn.decomposition import PCA
            # scale
            saber.log.info("pca scale...")
            pca = PCA(85, copy=False)
            pca.fit(all_scale)
            os.makedirs(os.path.dirname(cached_scale), exist_ok=True)
            with open(cached_scale, "wb") as fp:
                pickle.dump(dict(compT=pca.components_.T, means=pca.mean_), fp)
            scale_compT, scale_means = pca.components_.T, pca.mean_
            print('scale', pca.explained_variance_ratio_.cumsum()[-1])
            # print("scale=[" + ",".join([str(x) for x in pca.explained_variance_ratio_.cumsum()]) + "]")
            del pca

            # rotat
            saber.log.info("pca rotat...")
            pca = PCA(180, copy=False)
            pca.fit(all_rotat)
            os.makedirs(os.path.dirname(cached_rotat), exist_ok=True)
            with open(cached_rotat, "wb") as fp:
                pickle.dump(dict(compT=pca.components_.T, means=pca.mean_), fp)
            rotat_compT, rotat_means = pca.components_.T, pca.mean_
            print('rotat', pca.explained_variance_ratio_.cumsum()[-1])
            # print("rotat=[" + ",".join([str(x) for x in pca.explained_variance_ratio_.cumsum()]) + "]")
            del pca
            # quit()

        return scale_compT, scale_means, rotat_compT, rotat_means
