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
    text as text_utils
)
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from saberspeech.datasets.tools import get_features, data_info
from saberspeech.datasets.speech_anime import SpeechAnimeDataset


class DatasetWin2Frm(SpeechAnimeDataset):
    trainset_pca_compT = None
    trainset_pca_means = None

    def __init__(self, hparams: saber.ConfigDict, training: bool):
        super().__init__(hparams, training)
        # initialize hparams
        self.feat_names = deepcopy(hparams.audio.feature.feat_names)
        self.win_size = hparams.audio[self.feat_names[0].split("-")[0]].win_size
        self.hop_size = hparams.audio[self.feat_names[0].split("-")[0]].hop_size
        self.from_disk = hparams.audio.feature.from_disk
        self.feat_frames = hparams.audio.feature.sliding_window
        self.sliding_size = self.hop_size*(self.feat_frames-1)+self.win_size
        self.blend_type = hparams.anime.feature.blend_type
        # check
        assert isinstance(self.feat_names, (tuple, list))
        for name in self.feat_names:
            win_size = hparams.audio[name.split("-")[0]].win_size
            hop_size = hparams.audio[name.split("-")[0]].hop_size
            assert self.win_size == win_size, "feature's win_size not same!"
            assert self.hop_size == hop_size, "feature's hop_size not same!"

        self.maybe_calculate_phoneme_frequency()

        # set audio random shifting
        self.audio_shifting = math.floor(hparams.audio.sample_rate/hparams.anime.fps/2)

        # # ignore some data!
        # self.info_list = [
        #     info for info in self.info_list
        #     if not (info["speaker:str"] == "m4" and info["sentence_id:int"] == 29)
        # ]

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

            # # adjust start and end sample index according to anime ts
            # stt_ts = info["anime_mints:float"]
            # end_ts = info["anime_maxts:float"]
            # stt = math.ceil(self.ms_to_sample(stt_ts))
            # end = math.floor(self.ms_to_sample(end_ts))
            # stt_sp = max(stt_sp, stt - self.sliding_size // 2 - 1)
            # end_sp = min(end_sp, end + self.sliding_size // 2 + 1)

            # # get all frames
            # idx = 0
            # while self.frame_in_range(idx, self.sliding_size, stt_sp, end_sp):
            #     s = int(stt_sp + math.floor(self.frame_to_sample(idx)))
            #     e = int(s + self.sliding_size)
            #     self.coordinates.append({
            #         "data_id": i,
            #         "range": (s, e),
            #         "limit": (stt_sp, end_sp)
            #     })
            #     fid += 1
            #     idx += 1

        # transform all frames into pca
        if DatasetWin2Frm.trainset_pca_compT is None:
            assert self.training
            compT, means = self.pca()
            DatasetWin2Frm.trainset_pca_compT = compT
            DatasetWin2Frm.trainset_pca_means = means
        else:
            compT = DatasetWin2Frm.trainset_pca_compT
            means = DatasetWin2Frm.trainset_pca_means

        progress = saber.log.tqdm(self.info_list, desc="pca, projection")
        for data_dict in progress:
            data_dir = data_dict["npy_data_path:path"]
            info = data_info.get_info(data_dir)
            save_path = os.path.join(self.pca_path_prefix, info.speaker, info.emotion, str(info.sent_id).zfill(3) + ".npy")
            if os.path.exists(save_path):
                continue
            offsets_files = sorted(
                saber.filesystem.find_files(data_dir, r".*\.npy"),
                key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
            )
            progress.set_description(f"process {len(offsets_files)} frames")
            offsets = []
            for npy_path in offsets_files:
                offsets.append(np.load(npy_path))
            coeffs = np.asarray(offsets) - means
            coeffs = np.dot(coeffs, compT).astype(np.float32)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, coeffs)

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

        # get feature on-fly
        if self.training and (not self.from_disk):
            # choose signal
            rand_reverb = self.hparams.audio.feature.random_reverb
            rand_ps = self.hparams.audio.feature.random_pitch_shift
            rand_ts = self.hparams.audio.feature.random_time_stretch
            rand_noise = self.hparams.audio.feature.random_noise
            rand_preemph = self.hparams.audio.feature.random_preemph

            source_list = ["audio", "audio_denoised"]
            if rand_reverb: source_list.append("audio_reverb")
            if rand_ps: source_list.append("audio_ps")
            if rand_ts: source_list.append("audio_ts")
            source_type = np.random.choice(source_list)
            # print(source_type)
            if source_type in ["audio", "audio_reverb", "audio_denoised"]:
                signal = data[source_type]
            elif source_type == "audio_ps":
                suffix_list = ["-2", "2"]
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
                noise_type = np.random.choice(["white", "pink", "none"])
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
                weight = 1.0
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

        def get_part_energy(left, right):
            return librosa.feature.rms(
                y=signal[left: right],
                frame_length=self.win_size,
                hop_length=self.hop_size,
                center=False
            )

        ph_aligned = data.get("phonemes")
        feat0, ph0, wav0 = self._audio_features(signal, l0, r0, ph_aligned, training=self.training, **audio_feat_args)
        feat1, ph1, wav1 = self._audio_features(signal, l1, r1, ph_aligned, training=self.training, **audio_feat_args)

        ret = {
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

        # de_feat0, _, _ = self._audio_features(denoised, l0, r0, training=self.training, feat_extra=(0, ex_time))
        # de_feat1, _, _ = self._audio_features(denoised, l1, r1, training=self.training, feat_extra=(0, ex_time))
        # ret["denoised_audio_feat_0"] = de_feat0
        # ret["denoised_audio_feat_1"] = de_feat1

        symbol_type = self.hparams.text.type
        if symbol_type not in [None, "char"]:
            ret["phonemes_0"], ret["viseme_weight_0"] = ph0, 1.0
            ret["phonemes_1"], ret["viseme_weight_1"] = ph1, 1.0

        ret["anime_feat_0"], ret["anime_weight_0"] = get_anime(l0, r0)
        ret["anime_feat_1"], ret["anime_weight_1"] = get_anime(l1, r1)

        if self.hparams.anime.feature.anime_window > 0:
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
            feat, _, _ = cls._audio_features(part_wav)
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

        if stt is None: stt = 0
        if end is None: end = len(signal)

        feat, phs, wav = get_features.windowed_features(
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
        )

        if phs is not None and cls.hparams.text.type not in [None, "char"]:
            phs = np.asarray([
                saber.text.PhonemeNopad.convert_stress_to_typeid(ph, cls.hparams.text.type)
                for ph in phs
            ], dtype=np.int64)
        else:
            phs = None

        return feat, phs, wav

    @property
    def pca_path_prefix(self):
        return os.path.join(self.root, "pca", "_".join(os.path.splitext(x)[0] for x in self.csv_files))

    def pca(self):
        # using cached
        cache_path = self.pca_path_prefix + "_comp_97.pkg"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                cached = pickle.load(fp)
                compT, means = cached["compT"], cached["means"]
            saber.log.info("compT: {}, means: {}".format(compT.shape, means.shape))
        else:
            step = 1
            npy_path_list = []
            for data_dict in self.info_list:
                data_dir = data_dict["npy_data_path:path"]
                for i, npy_path in enumerate(saber.filesystem.find_files(data_dir, r".*\.npy")):
                    if step > 1 and i % step != 0:
                        continue
                    npy_path_list.append(npy_path)

            full_shape = (len(npy_path_list), len(np.load(npy_path_list[0])))
            all_verts = np.zeros(full_shape, dtype=np.float32)
            for r, npy_path in enumerate(saber.log.tqdm(npy_path_list, desc="pca, find frames")):
                all_verts[r] = np.load(npy_path)

            # pca
            from sklearn.decomposition import PCA, IncrementalPCA
            saber.log.info("pca...")
            n_comp = 50
            if full_shape[1] == 89784:
                n_comp = 160  # 290 for full dg, 160 for lower-face dg
            pca = PCA(n_components=n_comp, copy=False)
            with saber.log.timeit("pca fitting"):
                pca.fit(all_verts)
            print(pca.explained_variance_ratio_.cumsum()[-1], len(pca.explained_variance_ratio_))
            saber.log.info("compT: {}, means: {}".format(pca.components_.T.shape, pca.mean_.shape))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as fp:
                pickle.dump(dict(compT=pca.components_.T, means=pca.mean_), fp)
            compT, means = pca.components_.T, pca.mean_
        return compT, means

    def maybe_calculate_phoneme_frequency(self):
        # guard with no need
        phoneme_type = self.hparams.text.type
        if (phoneme_type not in ["stress_phoneme", "phoneme", "viseme"]):
            return

        # guard with preset
        info_key = "phoneme_information"
        phoneme_information = self.hparams.dataset_anime.get(info_key)
        if phoneme_information is not None:
            self.phoneme_information = deepcopy(phoneme_information)
            saber.log.info("'{}' is using phoneme information from hparams".format(
                "trainset" if self.training else "validset"
            ))
            return

        # collect phoneme_information
        total_count = dict(
            stress_phoneme=0,
            phoneme=0,
            viseme=0
        )
        counter = dict(
            stress_phoneme=defaultdict(lambda: 0),
            phoneme=defaultdict(lambda: 0),
            viseme=defaultdict(lambda: 0)
        )
        numbers = dict(
            stress_phoneme=text_utils.PhonemeNopad.num_stress_phonemes(),
            phoneme=text_utils.PhonemeNopad.num_phonemes(),
            viseme=text_utils.PhonemeNopad.num_visemes(),
        )
        labels = dict(
            stress_phoneme=list(text_utils.PhonemeNopad.stress_phonemes()),
            phoneme=list(text_utils.PhonemeNopad.phonemes()),
            viseme=list(text_utils.PhonemeNopad.visemes()),
        )

        for d_info in tqdm(self.info_list, desc="collect phoneme information"):
            data_path = d_info["npy_data_path:path"] + "_audio"
            with open(data_path, "rb") as fp:
                data = pickle.load(fp)
            full_phs = text_utils.PhonemeUtils.segment_phonemes(
                sr=data["sr"],
                all_aligned_phonemes=data["phonemes"],
                signal_start=0,
                signal_end=len(data["audio"]),
                win_size=self.win_size,
                hop_size=self.hop_size
            )
            full_pids = [text_utils.PhonemeNopad.convert_stress_to_typeid(ph, query_type="stress_phoneme") for ph in full_phs]
            pids = [text_utils.PhonemeNopad.convert_stress_to_typeid(ph, query_type="phoneme") for ph in full_phs]
            vids = [text_utils.PhonemeNopad.convert_stress_to_typeid(ph, query_type="viseme") for ph in full_phs]

            for i_ph in range(len(full_phs)):
                for key in total_count:
                    total_count[key] += 1
                counter["stress_phoneme"][full_pids[i_ph]] += 1
                counter["phoneme"][pids[i_ph]] += 1
                counter["viseme"][vids[i_ph]] += 1

        phoneme_information = {
            key: [
                dict(
                    freq=float(counter[key].get(pi, 0)) / float(total_count[key]),
                    weight=(
                        (float(total_count[key]) / float(counter[key].get(pi, 0) * numbers[key]))
                        if counter[key].get(pi, 0) > 0 else 0.0
                    ),
                    label=labels[key][pi]
                )
                for pi in range(numbers[key])
            ]
            for key in total_count
        }
        self.hparams.dataset_anime.set_key(info_key, phoneme_information)
        self.phoneme_information = deepcopy(self.hparams.dataset_anime.get(info_key))
