import os
import cv2
import math
import saber
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from .frame import render_frame
from ..tools import FaceDataType

_default_font = "saber/assets/Roboto-Regular.ttf"
roboto_font = ImageFont.truetype(_default_font, 32)


def put_texts(canvas, text_tuples, font=roboto_font):
    img_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img_pil)

    ascent, descent = font.getmetrics()
    height = ascent + descent
    for tup in text_tuples:
        txt, pos, color = tup
        (width, _), _ = font.font.getsize(txt)

        # get x, y
        x, y, aX, aY = pos
        if   aX == "left":   pass
        elif aX == "center": x = x - width // 2
        elif aX == "right":  x = x - width
        else: raise NotImplementedError()

        if   aY == "top":    pass
        elif aY == "middle": y = y - height // 2
        elif aY == "buttom": y = y - height
        else: raise NotImplementedError()

        draw.text((x, y), txt, font=font, fill=color)
    canvas = np.array(img_pil)
    return canvas


def render_video(
    sources:    list,
    video_fps:  int,
    audio_sr:   int,
    rows:       int  = 0,
    cols:       int  = 0,
    grid_w:     int  = 500,
    grid_h:     int  = 500,
    font_size:  int  = 32,
    save_video: bool = False,
    video_path: str  = None,
    verbose:    bool = True,
    per_frame:  bool = False,
    **kwargs
):
    _margin = 0
    _readers = {}
    _last_images = {}
    _background = 0
    _text_color = (220, 220, 200)
    _candidates = ["images", "video"] + list(FaceDataType.__members__)

    # get font
    tmp_font = ImageFont.truetype(_default_font, font_size)

    def _position(idx):
        r = idx // cols
        c = idx % cols
        x = c * (grid_w + _margin) + _margin
        y = r * (grid_h + _margin) + _margin
        return (x, y)

    def _render(key, src_dict, ts):
        source = src_dict.get(key)
        tslist = src_dict.get("tslist")
        # should be checked
        frame = saber.stream.seek(ts, tslist, source)
        # render
        if key == "images":
            return cv2.resize(frame, (grid_w, grid_h))
        else:
            return render_frame(frame, face_data_type=key, image_size=(grid_w, grid_h))

    def _read_video(key, ts, crop):
        img = _last_images.get(key, None)
        reader: cv2.VideoCapture = _readers[key]
        delta_ts = 1000.0 / float(reader.get(cv2.CAP_PROP_FPS))
        while reader.get(cv2.CAP_PROP_POS_MSEC) + delta_ts < ts or img is None:
            success, new_frame = reader.read()
            if success: img = new_frame
            else:       break
        img = img[:, :, :3]
        # resize image
        if img.shape[0] != grid_h or img.shape[1] != grid_w:
            ratio_w = grid_w / float(img.shape[1])
            ratio_h = grid_h / float(img.shape[0])
            if crop:
                ratio = max([ratio_w, ratio_h])
                dst_w = int(ratio * img.shape[1])
                dst_h = int(ratio * img.shape[0])
                img = cv2.resize(img, (dst_w, dst_h))
                crop_w = dst_w - grid_w
                crop_h = dst_h - grid_h
                ys, ye = crop_h//2, -(crop_h-crop_h//2)
                xs, xe = crop_w//2, -(crop_w-crop_w//2)
                if ye == 0: ye = img.shape[0]
                if xe == 0: xe = img.shape[1]
                img = img[ys: ye, xs: xe]
            else:
                ratio = min([ratio_w, ratio_h])
                dst_w = int(ratio * img.shape[1])
                dst_h = int(ratio * img.shape[0])
                img = cv2.resize(img, (dst_w, dst_h))
                pad_w = grid_w - dst_w
                pad_h = grid_h - dst_h
                img = np.pad(img, [[pad_h//2, pad_h-pad_h//2], [pad_w//2, pad_w-pad_w//2], [0, 0]],
                             "constant", constant_values=_background)
        # print(img.shape)
        _last_images[key] = img
        return img

    # check video or not
    vpath, apath = None, None
    dirname, rawname = None, None
    if save_video:
        assert video_path is not None, "'video_path' is not given!"
        dirname = os.path.dirname(video_path)
        rawname = os.path.splitext(os.path.basename(video_path))[0]
        vpath = os.path.join(dirname, "_{}.avi".format(rawname))
        apath = os.path.join(dirname, "_{}.wav".format(rawname))
    else:
        raise NotImplementedError("on-fly viewer is not implemented! 'save_video' must be True.")

    # analye sources and get an audio track
    sound_signal = None
    max_ts = float("-inf")
    render_sources = []
    for si, src_dict in enumerate(sources):
        # fetch audio track
        if (sound_signal is None) and ("audio" in src_dict):
            sound_signal = src_dict["audio"]
            if sound_signal.dtype == np.int16:
                sound_signal = sound_signal.astype(np.float32) / 32767.5
            elif sound_signal.dtype == np.float32:
                if not (sound_signal.min() >= -1 and sound_signal.max() <= 1):
                    saber.log.warn("signal (np.float32) is not in range (-1, 1).")
        # analyze source
        src_key = next((x for x in _candidates if x in src_dict), None)
        if src_key is not None:
            if src_key == "video":
                src_video = src_dict[src_key]
                assert os.path.exists(src_video), "failed to find video {}".format(src_video)
                _readers[src_video] = cv2.VideoCapture(src_video)
                _total_frames = _readers[src_video].get(cv2.CAP_PROP_FRAME_COUNT)
                _fps = _readers[src_video].get(cv2.CAP_PROP_FPS)
                max_ts = max(max_ts, _total_frames * 1000.0 / _fps)
            else:
                tslist = src_dict.get("tslist")
                assert tslist is not None,\
                    "'{}' is not given for '{}'".format(tslist, src_key)
                # update max_ts and count
                max_ts = max(max_ts, tslist[-1])
            render_sources.append(src_dict)
        else:
            msg = "unknown source at {}".format(si)
            if "title" in src_dict:
                msg += " of title '{}'".format(src_dict["title"])
            saber.log.warn(msg)
    # set sources
    sources = render_sources

    # no 'audio' in sources, get audio from 'video'
    if sound_signal is None:
        for si, src_dict in enumerate(sources):
            if "video" in src_dict:
                sound_signal = saber.audio.load(src_dict["video"], audio_sr)
                break

    # analyze layout
    if len(sources) == 0:
        raise ValueError("No valid render source is given!")
    if rows > 0 and cols <= 0:
        cols = int(math.ceil(len(sources) / rows))
    elif rows <= 0 and cols > 0:
        rows = int(math.ceil(len(sources) / cols))
    elif rows <= 0 and cols <= 0:
        # fully auto
        rows = int(math.floor(math.sqrt(len(sources))))
        cols = int(math.ceil(len(sources) / rows))

    canvas_size = (
        rows * grid_h + (rows + 1) * _margin,
        cols * grid_w + (cols + 1) * _margin,
        3
    )
    canvas = np.zeros(canvas_size, dtype=np.uint8)

    # render according to time sequence
    writer = None
    if vpath is not None:
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
        if per_frame:
            os.makedirs(video_path + "-frames/details", exist_ok=True)
        writer = cv2.VideoWriter()
        writer.open(
            vpath, cv2.VideoWriter_fourcc(*'XVID'), video_fps,
            (canvas_size[1], canvas_size[0])
        )

    # iterate all timestamps
    ts = 0
    delta_ts = 1000.0 / float(video_fps)
    total_frames = int(max_ts / delta_ts)
    if verbose:
        progress = saber.log.tqdm(
            range(total_frames),
            desc="'{}'".format(os.path.basename(video_path)),
            leave=False
        )
    while ts < max_ts:
        # render each source and fill into canvas
        index = 0
        canvas.fill(_background)
        txt_list = []
        for si, src_dict in enumerate(sources):
            # analyze source
            src_key = next((x for x in _candidates if x in src_dict), None)
            title = src_dict.get("title", "")

            # get frame image
            img = None
            ts_delay = src_dict.get("ts_delay", 0)

            # read from video
            if src_key == "video":
                img = _read_video(src_dict[src_key], ts - ts_delay, src_dict.get("crop", False))
                img = img[:, :, [2, 1, 0]]
            # render frame
            else:
                img = _render(src_key, src_dict, ts - ts_delay)

            # a valid frame
            if img is not None:
                # position index
                pi = src_dict.get("index", index)
                x, y = _position(pi)
                canvas[y: y+grid_h, x: x+grid_w, :3] = img[:, :, :3]
                # the text
                if len(title) > 0:
                    txt_pos = (x+grid_w//2, y+grid_h-20, "center", "buttom")
                    txt_list.append((title, txt_pos, _text_color))

                # dump each frame
                if per_frame:
                    frame_fname = video_path + f"-frames/details/frame-{int(ts)}-{index}.png"
                    if img.ndim == 4:
                        cv2.imwrite(frame_fname, img[:, :, [2, 1, 0, 3]])
                    elif img.ndim == 3:
                        cv2.imwrite(frame_fname, img[:, :, [2, 1, 0]])
                # update index
                index += 1

        # put texts
        canvas = put_texts(canvas, txt_list, font=tmp_font)
        # cv2.imshow('img', canvas)
        # cv2.waitKey(1)
        # write into video
        if writer is not None:
            writer.write(canvas[:, :, [2, 1, 0]])
        # dump each frame
        if per_frame:
            frame_fname = video_path + f"-frames/frame-{int(ts)}.png"
            cv2.imwrite(frame_fname, canvas[:, :, [2, 1, 0]])
        # next frame
        ts += delta_ts
        if verbose:
            progress.update()

    if verbose:
        progress.close()

    if writer is not None:
        writer.release()
        # ffmpeg merge audio
        saber.audio.save(apath, sound_signal, audio_sr)
        os.system("ffmpeg -loglevel panic -i '{}' -i '{}' -crf 15 -strict experimental '{}' -y".format(
            vpath, apath, video_path
        ))
        if os.path.exists(vpath): os.remove(vpath)
        if os.path.exists(apath): os.remove(apath)
        saber.log.info("write video into '{}'".format(video_path))

    # release all video captures
    for key in _readers:
        _readers[key].release()
