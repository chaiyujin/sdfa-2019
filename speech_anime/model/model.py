import os
import re
import cv2
import torch
import saber
import pickle
import librosa
import numpy as np
from copy import deepcopy
from .. import modules
from ..tools import FaceDataType, PredictionType
from .criterion import PLoss, MLoss, ELoss, DynamicLossScaler


class TwoPhase(torch.nn.Module):
    """ audio_feat -> anime_feat model """

    def __init__(self, hparams):
        super().__init__()
        self._audio_encoder = modules.Configurable(hparams, "audio_encoder")
        self._output_module = modules.OutputModule(hparams)
        if "speaker_embedding" in hparams.model:
            self._speaker_embedding = modules.SpeakerEmbedding(hparams)

    def forward(self, audio_feat, speaker_id):
        condition = None
        align_dict = dict()
        latent_dict = dict()
        if speaker_id is not None and hasattr(self, "_speaker_embedding"):
            condition = self._speaker_embedding(speaker_id)

        x = self._audio_encoder(
            audio_feat,
            condition=condition,
            align_dict=align_dict,
            latent_dict=latent_dict
        )
        preds = self._output_module(
            x,
            condition=condition,
            align_dict=align_dict,
            latent_dict=latent_dict
        )
        return dict(
            prediction=preds,
            condition=condition,
            align_dict=align_dict,
            latent_dict=latent_dict,
        )


class SaberTwoPhase(saber.SaberModel):
    """ Handle training, evaluation and other things """

    def __init__(self, hparams, trainset, validset):
        super().__init__(hparams, trainset, validset)
        self._model = TwoPhase(hparams)
        self._face_type = self._model._output_module._face_type
        self._pred_type = self._model._output_module._pred_type
        self._anime_loss_weight = hparams.loss.get("anime_loss_weight")
        # speakers, emotions
        self._speakers_dict = deepcopy(hparams.dataset_anime.speakers)
        self._emotions_dict = deepcopy(hparams.dataset_anime.emotions)
        # for losses
        self._loss_fns = dict()
        if self._face_type == FaceDataType.dgrad_3d:
            self._loss_fns["ploss_scale"] = PLoss(hparams)
            self._loss_fns["mloss_scale"] = MLoss(hparams)
            self._loss_fns["dyn_p_scale"] = DynamicLossScaler()
            self._loss_fns["dyn_m_scale"] = DynamicLossScaler()
            self._loss_fns["ploss_rotat"] = PLoss(hparams)
            self._loss_fns["mloss_rotat"] = MLoss(hparams)
            self._loss_fns["dyn_p_rotat"] = DynamicLossScaler()
            self._loss_fns["dyn_m_rotat"] = DynamicLossScaler()
        else:
            self._loss_fns["ploss"] = PLoss(hparams)
            self._loss_fns["mloss"] = MLoss(hparams)
            self._loss_fns["dyn_p"] = DynamicLossScaler()
            self._loss_fns["dyn_m"] = DynamicLossScaler()
        self._loss_fns["eloss"] = ELoss(hparams)
        self._loss_fns["dyn_e"] = DynamicLossScaler()

    def forward(self, batch):
        return self._model(
            audio_feat=batch["audio_feat"],
            speaker_id=batch["speaker_id"]
        )

    def train_step(self, batch, i_batch):
        self.train()
        pred_dict = self(batch)
        loss_dict, scalars = self.get_loss(pred_dict, batch)
        return pred_dict, loss_dict, scalars

    def valid_step(self, batch, i_batch):
        self.eval()
        pred_dict = self(batch)
        loss_dict, scalars = self.get_loss(pred_dict, batch)
        return pred_dict, loss_dict, scalars

    def evaluate(self, sources, experiment=None, in_trainer=True):
        from .. import viewer
        from ..datasets import DatasetSlidingWindow
        from ..tools.generate import prepare_sources_dict, _load_source, _append_images_source

        sources_dict = prepare_sources_dict(
            os.path.join(experiment.log_dir, "eval_at_train"),
            sources,
            overwrite_video=True
        )

        sr = self.hp.audio.sample_rate
        fps = self.hp.anime.fps

        # process all sources
        for _, sources in sources_dict.items():
            for src_args in sources:
                os.makedirs(os.path.dirname(src_args.output), exist_ok=True)
                name, ext = os.path.splitext(os.path.basename(src_args.path))
                true_data, signal, sound_signal = _load_source(src_args.path, sr, denoise_audio=False)
                if signal is None:
                    continue
                # to render
                render_list = []
                if true_data is not None:
                    render_list.append(true_data)
                # normalize singal
                signal = saber.audio.rms.normalize(
                    signal, self.hp.dataset_anime.audio_target_db)
                # predicate
                saber.log.info(f"infer from {name}")
                # predicate animation
                tslist, animes, others =\
                    self.generate_animation(
                        signal=signal,
                        dataset_class=DatasetSlidingWindow,
                        **src_args,
                    )
                # infer dict
                inferred = {
                    "title": f"infer: {name}",
                    "audio": sound_signal
                }
                face_type = self.hp.model.face_data_type
                inferred[face_type] = animes
                inferred["tslist"] = tslist
                # append to sources
                render_list.append(inferred)

                _append_images_source(render_list, sound_signal, others, "inputs", tslist)  # inputs
                _append_images_source(render_list, sound_signal, others, "latent", tslist)  # latent
                _append_images_source(render_list, sound_signal, others, "latent_align", tslist)
                _append_images_source(render_list, sound_signal, others, "formants", tslist)

                viewer.render_video(
                    sources=render_list,
                    video_fps=fps,
                    audio_sr=44100,
                    save_video=True,
                    video_path="{}/[{:04d}]{}".format(os.path.dirname(src_args.output), self.current_epoch, os.path.basename(src_args.output)),
                    grid_w=500,
                    grid_h=500,
                    font_size=24,
                )

    def data_to_anime_feat(self, tensor_dict, is_prediction):
        if self._pred_type == PredictionType.pca_normal:
            raise NotImplementedError()
        if self._pred_type == PredictionType.pca_coeffs:
            if self._face_type == FaceDataType.dgrad_3d:
                scale = tensor_dict["dgrad_3d_scale"]
                rotat = tensor_dict["dgrad_3d_rotat"]
                scale = self._model._output_module._scale_pca(scale)
                rotat = self._model._output_module._rotat_pca(rotat)
                shape_s = list(scale.size())[:-1] + [-1, 6]
                shape_r = list(rotat.size())[:-1] + [-1, 3]
                data = torch.cat((
                    scale.view(*shape_s),
                    rotat.view(*shape_r)
                ), dim=-1)
                shape = list(data.size())[:-2] + [-1]
                return data.view(*shape)
            else:
                coeff = tensor_dict[self._face_type.name + "_pca"]
                data = self._model._output_module._pca(coeff)
                return data
        if self._pred_type == PredictionType.face_data:
            if self._face_type == FaceDataType.dgrad_3d:
                scale = tensor_dict[self._face_type.name + "_scale"]
                rotat = tensor_dict[self._face_type.name + "_rotat"]
                shape_s = list(scale.size())[:-1] + [-1, 6]
                shape_r = list(rotat.size())[:-1] + [-1, 3]
                data = torch.cat((
                    scale.view(*shape_s),
                    rotat.view(*shape_r)
                ), dim=-1)
                shape = list(data.size())[:-2] + [-1]
                return data.view(*shape)
            else:
                return tensor_dict[self._face_type.name]

    def get_loss(self, pred_dict, batch):
        losses = dict()
        scalars = dict()
        hp = self.hp.loss
        preds = pred_dict["prediction"]
        bsz = batch["audio_feat"].size(0)
        device = batch["audio_feat"].device

        # get pred feat
        postfix = "_pca" if self._pred_type.name.find("pca") == 0 else ""

        # get anime loss weight
        anime_weight = (
            batch[self._anime_loss_weight]
            if self._anime_loss_weight is not None else
            torch.ones((bsz)).to(device)
        )

        # 2 branch
        if self._face_type == FaceDataType.dgrad_3d:
            pred_s = preds.get(f"dgrad_3d_scale{postfix}")
            pred_r = preds.get(f"dgrad_3d_rotat{postfix}")
            true_s = batch.get(f"dgrad_3d_scale{postfix}")
            true_r = batch.get(f"dgrad_3d_rotat{postfix}")
            # get loss
            ploss_s = self._loss_fns["ploss_scale"](pred_s, true_s, anime_weight)
            mloss_s = self._loss_fns["mloss_scale"](pred_s, true_s, anime_weight)
            ploss_r = self._loss_fns["ploss_rotat"](pred_r, true_r, anime_weight)
            mloss_r = self._loss_fns["mloss_rotat"](pred_r, true_r, anime_weight)
            scalars["scalar_ps"] = float(ploss_s.mean().item())
            scalars["scalar_ms"] = float(mloss_s.mean().item())
            scalars["scalar_pr"] = float(ploss_r.mean().item())
            scalars["scalar_mr"] = float(mloss_r.mean().item())
            scalars["scalar_ploss"] = scalars["scalar_ps"] + scalars["scalar_pr"]
            scalars["scalar_mloss"] = scalars["scalar_ms"] + scalars["scalar_mr"]
            if hp.dynamic_scalar:
                losses["dyn_ps"] = self._loss_fns["dyn_p_scale"].scale_loss(ploss_s, self.training) * float(hp.ploss_scale)
                losses["dyn_ms"] = self._loss_fns["dyn_m_scale"].scale_loss(mloss_s, self.training) * float(hp.mloss_scale)
                losses["dyn_pr"] = self._loss_fns["dyn_p_rotat"].scale_loss(ploss_r, self.training) * float(hp.ploss_scale)
                losses["dyn_mr"] = self._loss_fns["dyn_m_rotat"].scale_loss(mloss_r, self.training) * float(hp.mloss_scale)
            else:
                losses["loss_ps"] = ploss_s.mean() * float(hp.ploss_scale)
                losses["loss_ms"] = mloss_s.mean() * float(hp.mloss_scale)
                losses["loss_pr"] = ploss_r.mean() * float(hp.ploss_scale)
                losses["loss_mr"] = mloss_r.mean() * float(hp.mloss_scale)
        else:
            pred_anime = preds.get(f"{self._face_type.name}{postfix}")
            true_anime = batch.get(f"{self._face_type.name}{postfix}")
            # get loss
            ploss = self._loss_fns["ploss"](pred_anime, true_anime, anime_weight)
            mloss = self._loss_fns["mloss"](pred_anime, true_anime, anime_weight)
            scalars["scalar_ploss"] = float(ploss.mean().item())
            scalars["scalar_mloss"] = float(mloss.mean().item())
            if hp.dynamic_scalar:
                losses["dyn_ploss"] = self._loss_fns["dyn_p"].scale_loss(ploss, self.training) * float(hp.ploss_scale)
                losses["dyn_mloss"] = self._loss_fns["dyn_m"].scale_loss(mloss, self.training) * float(hp.mloss_scale)
            else:
                losses["loss_ploss"] = ploss.mean() * float(hp.ploss_scale)
                losses["loss_mloss"] = mloss.mean() * float(hp.mloss_scale)

        # evector
        if pred_dict.get("evector") is not None:
            eloss = self._loss_fns["eloss"](pred_dict["evector"])
            scalars["scalar_eloss"] = float(eloss.mean().item())
            if hp.dynamic_scalar:
                losses["dyn_eloss"] = self._loss_fns["dyn_e"].scale_loss(eloss, self.training) * float(hp.eloss_scale)
            else:
                losses["loss_eloss"] = eloss.mean() * float(hp.eloss_scale)

        return losses, scalars

    @torch.no_grad()
    def generate_animation(
        self, signal, speaker, emotion, frame_id,
        ensembling_ms=None, dataset_class=None,
        **kwargs
    ):
        # check signal
        if torch.is_tensor(signal):
            if signal.dim() > 1:
                assert signal.dim() == 2
                assert signal.size(0) == 1  # batch size must be 1!
                signal = signal[0]
            signal = signal.detach().cpu().numpy()
        assert isinstance(signal, np.ndarray)
        assert np.prod(signal.shape) == np.max(signal.shape)
        assert signal.min() >= -1
        assert signal.max() <= 1
        signal = signal.flatten()

        # check dataset class
        if dataset_class is None:
            dataset_class = self.trainset.__class__

        # check speaker and emotion
        if isinstance(speaker, str):
            speaker = self._speakers_dict[speaker]
        if isinstance(emotion, str):
            emotion = self._emotions_dict[emotion]

        if ensembling_ms is None:
            ensembling_ms = self.hp.ensembling_ms

        # cache of timestamps and feat_tuple
        if not hasattr(self, "_last_signal_data"):
            self._last_signal_data = None
            self._cached = None

        def _ensembling_audio(signal):
            # original
            features = dataset_class.fetch_audio_features(signal, self.hp)
            if ensembling_ms is None or ensembling_ms <= 0:
                return (features, )
            else:
                saber.log.warn("ensembling with {}ms".format(ensembling_ms))
                # get padding
                pad = ensembling_ms * self.hp.audio.sample_rate
                pad = pad // 1000
                # prev, post padding
                signal_prev = np.pad(signal[:-pad], [[pad, 0]], "constant")
                features_prev = dataset_class.fetch_audio_features(signal_prev, self.hp)
                # signal_post = np.pad(signal[pad:], [[0, pad]], "constant")
                # _, feat_post = dataset_class.fetch_audio_features(signal_post, self.hp)
                return (features, features_prev)

        def _ensembling_anime(features_tuple):
            _args = dict(
                speaker_id=speaker,
                emotion_id=emotion,
                frame_id=frame_id
            )
            anime_sum, others = self._feature_to_anime(
                feat_list=features_tuple[0]["audio_feat"],
                energy_list=features_tuple[0]["energy"],
                **_args
            )
            for i in range(1, len(features_tuple)):
                anime_sum += self._feature_to_anime(
                    feat_list=features_tuple[i]["audio_feat"],
                    energy_list=features_tuple[i]["energy"],
                    **_args
                )[0]
            return anime_sum / float(len(features_tuple)), others

        def _filter_anime(anime_sequence):
            return anime_sequence

        # get timestamps and audio features
        if (self._last_signal_data is not None) and (self._last_signal_data == signal.data):
            # cached
            features_tuple = self._cached
        else:
            features_tuple = _ensembling_audio(signal)
            del self._cached  # release
            self._cached = features_tuple
            self._last_signal_data = signal.data

        # return timestamps and anime frames
        raw_anime, others = _ensembling_anime(features_tuple)
        return features_tuple[0]["tslist"], _filter_anime(raw_anime), others

    def _ndarray_to_tensor(self, feat_list):
        feats = torch.FloatTensor(feat_list)
        if self.on_gpu:
            feats = feats.cuda()
        return feats

    def _feature_to_anime(
        self, feat_list, energy_list, speaker_id, emotion_id, frame_id, bs=100
    ):

        def _int_to_tensor(idx, size):
            assert isinstance(idx, int), f"given index is {idx}, {type(idx)}"
            idx = torch.LongTensor([idx] * size)
            if self.on_gpu:
                idx = idx.cuda()
            return idx

        self.eval()
        with torch.no_grad():
            eid = _int_to_tensor(emotion_id, bs)
            fid = _int_to_tensor(frame_id, bs)
            spk_id = _int_to_tensor(speaker_id, bs)
            feat_list = self._ndarray_to_tensor(feat_list)
            energy_list = self._ndarray_to_tensor(energy_list)
            animes = []
            inputs = []
            phones = []
            latent = []
            formants = []
            latent_align = []
            for i in saber.log.tqdm(range(0, len(feat_list), bs), desc="infer anime"):
                j = min(i + bs, len(feat_list))
                if j - i != bs:  # last one
                    eid = _int_to_tensor(emotion_id, j-i)
                    fid = _int_to_tensor(frame_id, j-i)
                    spk_id = _int_to_tensor(speaker_id, j-i)
                res = self.forward({
                    "audio_feat": feat_list[i: j],
                    "speaker_id": spk_id,
                    "emotion_id": eid,
                    "frame_id": fid,
                })
                _animes = self.data_to_anime_feat(res["prediction"], is_prediction=True)
                _inputs = feat_list[i: j]
                _animes = _animes.squeeze(1)  # T is 1
                _inputs = _inputs.permute(0, 3, 2, 1)
                animes.extend(_animes.detach().cpu().numpy())
                inputs.extend(_inputs.detach().cpu().numpy())
            # return anime and latent
            animes = np.asarray(animes, dtype=np.float32)

            def to_ndarray(arr):
                if len(arr) == 0:
                    return None
                return np.asarray(arr, np.float32)

            others = {
                "inputs": to_ndarray(inputs),
                "phones": to_ndarray(phones),
                "latent": to_ndarray(latent),
                "latent_align": to_ndarray(latent_align),
                "formants": to_ndarray(formants),
            }

            del eid
            del fid
            del feat_list
            del energy_list

        return animes, others
