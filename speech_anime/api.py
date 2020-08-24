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
        nb_train_log     = 1
    )

    trainer = saber.Trainer(exp)
    trainer.train()


def evaluate_model(args):
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

    model = model_class(hparams, trainset=None, validset=None)
    sources_dict = hparams.trainer.evaluate

    exp = saber.Experiment(model, hparams, hparams.log_dir, training=False)
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
        output_dir=args.output_dir or os.path.join(hparams.log_dir, "evaluate_videos"),
    )
