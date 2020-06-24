import os
from saber import filesystem
from collections import namedtuple

__all_emotions__ = ["neutral", "happy", "surprised", "angry", "sad"]
DataInfo = namedtuple("DataInfo", ["root", "speaker", "emotion", "sent_id"])


def get_info(path):
    sent_id = int(os.path.basename(path))
    emotion = os.path.basename(filesystem.ancestor(path, 1))
    speaker = os.path.basename(filesystem.ancestor(path, 2))
    root = filesystem.ancestor(path, 4)
    assert os.path.basename(filesystem.ancestor(path, 3)) == "data"
    assert emotion in __all_emotions__,\
        "Emotion '{}' is unknown! Should be in {}".format(emotion, __all_emotions__)
    return DataInfo(
        root    = root,
        speaker = speaker,
        emotion = emotion,
        sent_id = sent_id
    )


def get_path(root, speaker, emotion, sent_id, subdir="data"):
    return os.path.join(
        root.strip(),
        subdir,
        speaker.strip(),
        emotion.strip(),
        str(int(sent_id)).zfill(3)
    )


def has_emotion(path: str, emotion: str):
    info = get_info(path)
    return emotion == info.emotion


def possible_emotions():
    return __all_emotions__
