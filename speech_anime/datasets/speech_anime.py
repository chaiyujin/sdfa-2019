import os
import math
import torch
import saber
import numpy as np
from copy import deepcopy
from torch.utils.data.dataloader import default_collate


def _check_same_meta(meta_a, meta_b):
    assert len(meta_a) == len(meta_b),\
        "different meta:\n{}\n{}".format(meta_a, meta_b)
    for meta in meta_a:
        assert meta in meta_b,\
            "different meta:\n{}\n{}".format(meta_a, meta_b)


class SpeechAnimeDataset(torch.utils.data.dataset.Dataset):
    hparams = None
    __all_speakers__ = None
    __all_emotions__ = None
    __rev_speakers__ = None
    __rev_emotions__ = None

    def __init__(self, hparams: saber.ConfigDict, training: bool):
        # dataset configuration
        if SpeechAnimeDataset.hparams is None:
            SpeechAnimeDataset.hparams = hparams
            SpeechAnimeDataset.__all_speakers__ = hparams.dataset_anime.speakers
            SpeechAnimeDataset.__all_emotions__ = hparams.dataset_anime.emotions
            SpeechAnimeDataset.__rev_speakers__ = {sid: spk for spk, sid in hparams.dataset_anime.speakers.items()}
            SpeechAnimeDataset.__rev_emotions__ = {eid: emo for emo, eid in hparams.dataset_anime.emotions.items()}
        else:
            assert SpeechAnimeDataset.hparams is hparams,\
                "SpeechAnimeDataset should be create with same 'hparams'!"

        super().__init__()
        # training flag
        self.training = training
        # analyze csv files
        self.root = hparams.dataset_anime.root
        self.primary_key = hparams.dataset_anime.primary_key
        self.csv_files = (
            hparams.dataset_anime.train_list if training else
            hparams.dataset_anime.valid_list
        )
        self.meta_data = None
        self.info_list = list()
        self.primary_history = dict()
        for csv_file in self.csv_files:
            csv_file = os.path.join(self.root, csv_file)
            meta_data, info_list = saber.csv.read_csv(csv_file)
            if self.meta_data is None:
                self.meta_data = meta_data
            _check_same_meta(self.meta_data, meta_data)
            for info in info_list:
                # check speaker and emotion
                if (
                    info["speaker:str"] not in hparams.dataset_anime.speakers or
                    info["emotion:str"] not in hparams.dataset_anime.emotions
                ):
                    continue
                val = info[self.primary_key]
                if val not in self.primary_history:
                    self.primary_history[val] = True
                    self.info_list.append(info)
        # get speakers and emotions
        speakers = sorted(list(set(info["speaker:str"] for info in self.info_list)))
        emotions = sorted(list(set(info["emotion:str"] for info in self.info_list)))
        self.speakers = {spk: hparams.dataset_anime.speakers[spk] for spk in speakers}
        self.emotions = {emo: hparams.dataset_anime.emotions[emo] for emo in emotions}
        # default collate
        self.default_collate = default_collate

    @property
    def num_speakers(self):
        return len(self.speakers)

    @property
    def num_emotions(self):
        return len(self.emotions)

    @property
    def num_all_speakers(self):
        return len(self.__all_speakers__)

    @property
    def num_all_emotions(self):
        return len(self.__all_emotions__)

    def collate(self, batch):
        """ This function should collate batch into tensors.
            self.default_collate can be used.
        """
        raise NotImplementedError("'collate' is not implemented!")

    @classmethod
    def get_speaker_id(cls, speaker: str):
        try:
            return cls.__all_speakers__[str(speaker)]
        except KeyError:
            raise KeyError("speaker '{}' is not in {}".format(speaker, list(cls.__all_speakers__.keys())))

    @classmethod
    def get_emotion_id(cls, emotion: str):
        try:
            return cls.__all_emotions__[str(emotion)]
        except KeyError:
            raise KeyError("emotion '{}' is not in {}".format(emotion, list(cls.__all_emotions__.keys())))

    @classmethod
    def get_speaker_name(cls, speaker_id: int):
        return cls.__rev_speakers__[int(speaker_id)]

    @classmethod
    def get_emotion_name(cls, emotion_id: int):
        return cls.__rev_emotions__[int(emotion_id)]

    @classmethod
    def fetch_audio_features(cls, signal, **kwargs):
        """ This function should return audio features
            for generation
        """
        raise NotImplementedError("classmethod 'fetch_audio_features' is not implemented!")

    """ units """

    @classmethod
    def ms_to_sample(cls, ms, sr: int = None, dtype = np.float32):
        sr = sr or cls.hparams.audio.sample_rate
        sample = float(ms * sr) / 1000.0
        return dtype(sample)

    @classmethod
    def sample_to_ms(cls, sample, sr: int = None, dtype = np.float32):
        sr = sr or cls.hparams.audio.sample_rate
        ms = float(sample * 1000.0) / float(sr)
        return dtype(ms)

    @classmethod
    def frame_to_sample(cls, idx, sr: int = None, fps: int = None, dtype = np.float32):
        sr = sr or cls.hparams.audio.sample_rate
        fps = fps or cls.hparams.anime.fps
        sample = float(idx * sr) / float(fps)
        return dtype(sample)

    @classmethod
    def sample_to_frame(cls, sample, sr: int = None, fps: int = None, dtype = np.float32):
        sr = sr or cls.hparams.audio.sample_rate
        fps = fps or cls.hparams.anime.fps
        frame = float(sample * fps) / float(sr)
        return dtype(frame)

    @classmethod
    def frame_to_ms(cls, idx, fps: int = None, dtype = np.float32):
        fps = fps or cls.hparams.anime.fps
        ms = float(idx * 1000.0) / float(fps)
        return dtype(ms)

    @classmethod
    def ms_to_frame(cls, ms, fps: int = None, dtype = np.float32):
        fps = fps or cls.hparams.anime.fps
        frame = float(ms * fps) / 1000.0
        return dtype(frame)
