import os
import torch
import saber
import saberspeech
import multiprocessing
from .tools import configure, generate


def train_model(args):
    hparams, class_dict = configure(args)
    model_class = class_dict["model"]
    dataset_class = class_dict["dataset"]

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
            possible_roots=[check_root],
            possible_exts=[".ckpt"],
            must_be_found=True,
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
    hparams, class_dict = configure(args)

    # if load from is given
    if hparams.get("load_from") is not None:
        check_root = os.path.join(hparams.log_dir, "checkpoints")
        hparams.set_key("load_from", saber.filesystem.maybe_in_dirs(
            hparams.load_from,
            possible_roots=[check_root],
            possible_exts=[".ckpt"],
            must_be_found=True,
        ))

    dataset_class = class_dict["dataset"]
    model = class_dict["model"](hparams, trainset=None, validset=None)

    output_dir = os.path.join(hparams.log_dir, "evaluate_videos")
    sources_dict = hparams.trainer.evaluate

    exp = saber.Experiment(model, hparams, hparams.log_dir, training=False)

    exp.saber_model.eval()
    generate(
        exp.saber_model,
        output_dir,
        sources_dict,
        dataset_class
    )


def generate_from_sources(
    output_dir: str,
    sources_dict: dict,
    exp_dir,
    exp_ckpt,
    exp_hparams="hparams",
    **kwargs
):

    hparams, class_dict = configure(dict(
        mode           = "generate",
        log_dir        = exp_dir,
        load_from      = exp_ckpt,
        custom_hparams = exp_hparams,
        save_video     = True,
    ))

    model = class_dict["model"](hparams, trainset=None, validset=None)
    dataset_class = class_dict["dataset"]

    exp = saber.Experiment(model, hparams, training=False)
    exp.saber_model.eval()
    generate(
        exp.saber_model,
        output_dir,
        sources_dict,
        dataset_class,
        **kwargs
    )
