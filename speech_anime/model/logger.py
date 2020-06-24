import math
import saber
import torch
import numpy as np
from saber import (
    visualizer,
    audio as audio_utils
)
from ..tools import FaceDataType
from .. import viewer
plot_item = visualizer.plot_item


@saber.Experiment.register_plot
def plot_anime_frames(exp: saber.Experiment, num, tag, preds, batch):

    # guard
    if preds.get("prediction") is None:
        return

    pred_data = exp.saber_model.data_to_anime_feat(preds["prediction"], is_prediction=True)
    true_data = exp.saber_model.data_to_anime_feat(batch, is_prediction=False)
    for i_log in range(num):
        pred_anime = pred_data[i_log]
        true_anime = true_data[i_log]
        if FaceDataType.is_mesh(exp.saber_model._face_type):
            pred_verts, faces = viewer.frame_to_mesh(pred_anime, exp.saber_model._face_type)
            true_verts, faces = viewer.frame_to_mesh(true_anime, exp.saber_model._face_type)
            pred_verts = np.reshape(pred_verts, (1, -1, 3))
            true_verts = np.reshape(true_verts, (1, -1, 3))
            faces = np.reshape(faces, (1, -1, 3))
            exp.add_mesh(f"{tag}-{i_log}/frame-pred", pred_verts, faces=faces)
            exp.add_mesh(f"{tag}-{i_log}/frame-true", true_verts, faces=faces)


@saber.Experiment.register_plot
def plot_audio_features(exp: saber.Experiment, num, tag, preds, batch):

    # guard
    if preds.get("prediction") is None:
        return

    def _plot_align(i, tag):
        # guard
        if preds.get("align_dict") is None:
            return

        # plot attentions
        for key, align_batch in preds["align_dict"].items():
            to_plot = []
            align = align_batch[i].squeeze()
            assert 1 <= align.dim() <= 2
            to_plot.append(plot_item(align, "{}".format(key)))
            img = visualizer.plot(*to_plot, val_mode="auto")
            exp.add_image(tag + key, img)

    def _prepare_items(tensors, titles, title_postfix):
        ret = []
        for ch in range(tensors[0].shape[-1]):  # channels
            row = []
            for title, tensor in zip(titles, tensors):
                sub_tensor = tensor[..., ch].transpose(1, 0)
                row.append(plot_item(sub_tensor, "{}-{}(ch{})".format(title, title_postfix, ch)))
            ret.append(row)
        return ret

    def _audio_feats_to_img(subtag_list, tensor_list):
        if not isinstance(subtag_list, (list, tuple)):
            subtag_list = [subtag_list]
        if not isinstance(tensor_list, (list, tuple)):
            tensor_list = [tensor_list]
        # get info
        feat_name = exp.hparams.audio.feature.name
        to_plot = []
        to_plot.extend(_prepare_items(tensor_list, subtag_list, title_postfix=feat_name))
        img = visualizer.plot(*to_plot, val_mode="auto")
        return img

    def _plot_feats(i, tag):
        inputs = batch.get("audio_feat")
        if inputs is not None:
            subtags = ["inputs"]
            tensors = [inputs[i]]
            img = _audio_feats_to_img(subtags, tensors)
            exp.add_image(f"{tag}inputs-{i:02d}", img)

            bsz = inputs.size(0)
            subtags = ["inputs-adj"]
            tensors = [inputs[i+bsz//2]]
            img = _audio_feats_to_img(subtags, tensors)
            exp.add_image(f"{tag}inputs-{i+bsz//2:02d}", img)

    def _add_audio(i, tag):
        # guard
        if preds.get("prediction") is None:
            return
        signal = batch["signal"][i]
        exp.add_audio(tag, signal, sample_rate=int(batch["sr"][i]))

    for i_log in range(num):
        _plot_feats (i_log, f"{tag}-{i_log}/0.audio-feature.")
        _plot_align (i_log, f"{tag}-{i_log}/1.audio-align.")
        _add_audio  (i_log, f"{tag}-{i_log}/0.audio-signal.")
