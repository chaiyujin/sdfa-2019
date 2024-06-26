import os
import re
import cv2
import saber
import pickle
import librosa
import numpy as np


def _prepare_sources_dict(
    output_dir, sources_dict,
    overwrite_video=False,
    all_args=["path", "output", "speaker", "emotion", "frame_id"],
    defaults=[                  0,         0,         0],
    abbrs=dict(out="output", spk="speaker", emo="emotion"),
    **kwargs
):
    sources_dict = dict(sources_dict)
    # extend sources
    for src_type in sources_dict:
        sources = sources_dict[src_type]
        assert isinstance(sources, (tuple, list)),\
            f"sources_dict[{src_type}] is not a list or tuple!"
        sources_args_list = []

        for src_args in sources:
            if isinstance(src_args, str): src_args = [src_args]

            name = os.path.splitext(os.path.basename(src_args[0]))[0]
            output = os.path.join(output_dir, name + ".mp4")
            _defaults = [output] + defaults

            src_args = saber.utils.ArgumentParser(
                *src_args,
                all_args=all_args,
                defaults=_defaults,
                key_abbrs=abbrs
            )

            if not os.path.exists(src_args.output) or overwrite_video:
                sources_args_list.append(src_args)
            else:
                saber.log.warn(f"Output exists: {src_args.output}")

        sources_dict[src_type] = sources_args_list

    return sources_dict


def _load_source(path, sr, denoise_audio):
    path = os.path.expanduser(path)
    name, ext = os.path.splitext(os.path.basename(path))
    true_data = None
    signal, sound_signal = None, None
    if re.match(r"^\d+$", name):
        with open(path + "_audio", "rb") as fp:
            data = pickle.load(fp)
            signal = data["audio"]
            sound_signal = librosa.resample(signal, sr, 44100)
            assert sr == data["sr"]
        frames = []
        tslist = []
        npy_files = saber.filesystem.find_files(path, r"-*\d+\.npy")
        for fi, npy_path in enumerate(npy_files):
            frames.append(np.load(npy_path))
            tslist.append(fi * 1000.0 / fps)
        true_data = {
            "title": "true: {}".format(name),
            "audio": sound_signal,
            "verts_off_3d": frames,
            "tslist": tslist
        }
    elif ext == ".wav":
        sound_signal = saber.audio.load(path, 44100)
        if denoise_audio:
            saber.log.info("denoise audio")
            sound_signal = saber.audio.denoise(sound_signal, 44100)
        signal = librosa.resample(sound_signal, orig_sr=44100, target_sr=sr)
    elif ext in [".mp4", ".m4v", ".avi"]:
        sound_signal = saber.audio.load(path, 44100)
        if denoise_audio:
            saber.log.info("denoise audio")
            sound_signal = saber.audio.denoise(sound_signal, 44100)
        signal = librosa.resample(sound_signal, orig_sr=44100, target_sr=sr)
        true_data = {
            "title": "true: {}".format(name),
            "video": path
        }
    else:
        saber.log.warn("{} is not supported!".format(ext))
    return true_data, signal, sound_signal


def _append_images_source(render_list, sound_signal, others, name, tslist):
    if (name not in others
            or others[name] is None
            or len(others[name]) == 0):
        saber.log.warn("'{}' is not valid image source!".format(name))
        return
    # color mapping
    tensor = others[name]
    vmin = np.min(tensor)
    vmax = np.max(tensor)
    if others[name].ndim == 4:
        new_shape = (tensor.shape[0], -1, tensor.shape[3])
        tensor = np.reshape(tensor, new_shape)
    images = np.asarray([
        cv2.resize(
            saber.visualizer.color_mapping(inputs, vmin=vmin, vmax=vmax, flip_rows=True),
            (500, 500)
        )
        for inputs in saber.log.tqdm(tensor, desc="cmap")
    ], dtype=np.uint8)
    # source dict
    img_src = {
        "title": "{} ({:.2f}~{:.2f})".format(name, vmin, vmax),
        "audio": sound_signal,
        "images": images,
        "tslist": tslist
    }
    render_list.append(img_src)
