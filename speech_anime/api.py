from typing import Any

import os
import torch
import saber
import multiprocessing
from .tools import configure
from .model import SaberSpeechDrivenAnimation
from .datasets import DatasetSlidingWindow


def train_model(args):
    hparams = configure(args)
    model_class   = SaberSpeechDrivenAnimation
    dataset_class = DatasetSlidingWindow

    assert hparams.get("tag") is not None
    # get missing args from hparams
    if hparams.get("log_dir") is None:
        auto_dir = "experiments/results/[{}]{}".format(hparams.date_string(), hparams.tag)
        hparams.set_key("log_dir", auto_dir)

    # if load from is given
    if hparams.get("load_from") is not None:
        check_root = os.path.join(hparams.log_dir, "checkpoints")
        hparams.set_key("load_from", saber.filesystem.maybe_in_dirs(
            hparams.load_from,
            possible_roots = [check_root],
            possible_exts  = [".ckpt"],
            must_be_found  = True,
        ))

    train_loaders = dict()
    valid_loaders = dict()
    # create anime loaders
    anime_trainset = None
    if hparams.dataset_anime is not None:
        anime_trainset = dataset_class(hparams, True)
        anime_validset = dataset_class(hparams, False)
        num_workers = (
            multiprocessing.cpu_count() // 2
            if hparams.trainer.anime_loader.multiple_workers else
            0
        )
        train_loaders["anime"] = torch.utils.data.dataloader.DataLoader(
            anime_trainset, hparams.trainer.anime_loader.batch_size,
            collate_fn=anime_trainset.collate, drop_last=False,
            shuffle=True, num_workers=num_workers
        )
        valid_loaders["anime"] = torch.utils.data.dataloader.DataLoader(
            anime_validset, hparams.trainer.anime_loader.batch_size,
            collate_fn=anime_validset.collate, drop_last=False,
            shuffle=True, num_workers=num_workers
        )

    # create model, pca need trainset
    model = model_class(
        hparams=hparams,
        trainset=anime_trainset,
        validset=anime_validset
    )

    # create experiment
    exp = saber.Experiment(
        model, hparams,
        log_dir          = hparams.log_dir,
        train_loaders    = train_loaders,
        valid_loaders    = valid_loaders,
        main_loader_name = hparams.trainer.get("main_loader_name", "anime"),
        nb_train_log     = 1,
        ckpt_preprocess  = ckpt_backward_compatible_preprocess,
    )

    trainer = saber.Trainer(exp)
    trainer.train()


def evaluate_model(args):
    hparams = configure(args)
    model_class   = SaberSpeechDrivenAnimation
    dataset_class = DatasetSlidingWindow

    if hparams.eval_input is not None:
        new_record = [hparams.eval_input]
        if hparams.eval_spk_cond is not None:
            new_record.append(f"speaker={hparams.eval_spk_cond}")
        hparams.trainer.evaluate.set_key("test", [new_record])

    # if load from is given
    if hparams.get("load_from") is not None:
        check_root = os.path.join(hparams.log_dir, "checkpoints")
        hparams.set_key("load_from", saber.filesystem.maybe_in_dirs(
            hparams.load_from,
            possible_roots = [check_root],
            possible_exts  = [".ckpt"],
            must_be_found  = True,
        ))

    model = model_class(hparams, trainset=None, validset=None, load_pca=False)
    sources_dict = hparams.trainer.evaluate

    exp = saber.Experiment(model, hparams, hparams.log_dir, training=False, ckpt_preprocess=ckpt_backward_compatible_preprocess)

    # trace
    if args.get("traced_dump_path") is not None:
        exp.saber_model._model.eval()
        # trace on gpu
        audio_feat = torch.rand(1, 64, 128, 3, device="cuda:0")
        speaker_id = torch.zeros(1, dtype=torch.long, device="cuda:0")
        traced_model = torch.jit.trace(exp.saber_model._model, (audio_feat, speaker_id))
        traced_model.save(os.path.splitext(args.traced_dump_path)[0] + "-gpu.zip")
        exp.saber_model._traced_model = traced_model
        # trace on cpu
        audio_feat = torch.rand(1, 64, 128, 3, device="cpu")
        speaker_id = torch.zeros(1, dtype=torch.long, device="cpu")
        traced_model = torch.jit.trace(exp.saber_model._model.cpu(), (audio_feat, speaker_id))
        traced_model.save(os.path.splitext(args.traced_dump_path)[0] + "-cpu.zip")

    exp.saber_model.evaluate(
        sources_dict,
        experiment=None,
        in_trainer=False,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        font_size=args.font_size,
        with_title=args.with_title,
        draw_truth=args.draw_truth,
        draw_align=args.draw_align,
        draw_latent=args.draw_latent,
        overwrite_video=args.overwrite_video,
        export_mesh_frames=args.export_mesh_frames,
        output_dir=args.output_dir or os.path.join(hparams.log_dir, "evaluate_videos"),
    )


def jit_trace(args):
    hparams = configure(args)
    model_class   = SaberSpeechDrivenAnimation
    dataset_class = DatasetSlidingWindow

    # if load from is given
    if hparams.get("load_from") is not None:
        check_root = os.path.join(hparams.log_dir, "checkpoints")
        hparams.set_key("load_from", saber.filesystem.maybe_in_dirs(
            hparams.load_from,
            possible_roots = [check_root],
            possible_exts  = [".ckpt"],
            must_be_found  = True,
        ))

    assert args.get("traced_dump_path") is not None

    model = model_class(hparams, trainset=None, validset=None, load_pca=False)
    exp = saber.Experiment(model, hparams, hparams.log_dir, training=False, ckpt_preprocess=ckpt_backward_compatible_preprocess)
    exp.saber_model._model.eval()

    # trace on gpu
    audio_feat = torch.rand(1, 64, 128, 3, device="cuda:0")
    speaker_id = torch.zeros(1, dtype=torch.long, device="cuda:0")
    traced_model = torch.jit.trace(exp.saber_model._model, (audio_feat, speaker_id))
    traced_model.save(os.path.splitext(args.traced_dump_path)[0] + "-gpu.zip")

    # trace on cpu
    audio_feat = torch.rand(1, 64, 128, 3, device="cpu")
    speaker_id = torch.zeros(1, dtype=torch.long, device="cpu")
    traced_model = torch.jit.trace(exp.saber_model._model.cpu(), (audio_feat, speaker_id))
    traced_model.save(os.path.splitext(args.traced_dump_path)[0] + "-cpu.zip")


def ckpt_backward_compatible_preprocess(ckpt: Any):
    state_dict = ckpt["state"]
    # Backward compatible.
    replacement = (
        ("_ext_batch_norm", "_ext_post_bn"),
        ("audio_encoder.layers.0", "_model._audio_encoder._layers.1"),
        ("audio_encoder.layers.1", "_model._audio_encoder._layers.2"),
        ("audio_encoder.layers.2", "_model._audio_encoder._layers.3"),
        ("audio_encoder.layers.3", "_model._audio_encoder._layers.4"),
        ("audio_encoder.layers.4", "_model._audio_encoder._layers.5"),
        ("audio_encoder.layers.5", "_model._audio_encoder._layers.6"),
        ("time_aggregator.layers.0", "_model._audio_encoder._layers.9"),
        ("time_aggregator.layers.1", "_model._audio_encoder._layers.10"),
        ("anime_decoder.layers.", "_model._output_module._layers."),
        ("anime_decoder.layers_scale", "_model._output_module._scale_layers"),
        ("anime_decoder.layers_rotat", "_model._output_module._rotat_layers"),
        ("anime_decoder.proj_scale", "_model._output_module._scale_pca"),
        ("anime_decoder.proj_rotat", "_model._output_module._rotat_pca"),
    )
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k
        for on, nn in replacement:
            nk = nk.replace(on, nn)
        new_state_dict[nk] = v
    new_state_dict.pop("hamm")
    ckpt["state"] = new_state_dict
    return ckpt
