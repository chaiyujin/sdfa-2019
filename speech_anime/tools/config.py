import os
import saber
import torch
import random
import argparse
import importlib
import matplotlib
import numpy as np
import saber.utils.filesystem as saber_fs
from .. import viewer

__root__ = os.path.abspath(saber_fs.ancestor(__file__, level=2))


def configure(args) -> saber.ConfigDict:
    """ Configure the default hparams, maybe overwrite by args or other hparams file.
        Besides, torch cudnn state and all random seeds will be set.
    """
    # check inputs
    if isinstance(args, (str, dict, argparse.Namespace)):
        args = saber.ConfigDict(args)
    assert isinstance(args, saber.ConfigDict)
    args.check_keys("mode")

    config_root = os.path.join(__root__, "config")
    # load default hparams
    hparams = saber.ConfigDict(os.path.join(config_root, "default.py"))

    # check mode in args
    mode = args.get("mode")

    # find custom hparams
    if args.get("custom_hparams") is not None:
        filename = saber_fs.maybe_in_dirs(
            args.custom_hparams,
            must_be_found  = True,
            possible_exts  = [".json", ".py"],
            possible_roots = [os.path.join(config_root, "model"),
                              (args.get("log_dir") or ".")],
        )
        # load custom hparams
        custom = saber.ConfigDict(filename)
        if args.mode == "evaluate" and "evaluate" in custom.get("trainer", {}):
            del custom.trainer["evaluate"]
        # overwrite
        hparams.overwrite_by(custom)

    # set dataset anime
    _maybe_load_dataset_hparams("dataset_anime",  args, hparams)
    _maybe_load_dataset_hparams("dataset_speech", args, hparams)

    # overwrite hparams from args
    _overwrite_hparams(hparams, args, "tag")
    _overwrite_hparams(hparams, args, "seed")
    _overwrite_hparams(hparams, args, "log_dir")
    _overwrite_hparams(hparams, args, "load_from")
    _overwrite_hparams(hparams, args, "ensembling_ms")

    # matplotlib use
    matplotlib.use(hparams.get("matplotlib_use", "Agg"))

    # cudnn benchmark and deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set random seed
    seed = hparams.get("seed", 1234)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set default template
    default_template_mesh = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "datasets/vocaset/template/FLAME_sample.obj"
    )
    viewer.set_template_mesh(default_template_mesh)

    return hparams


def _overwrite_hparams(hparams, args, key):
    if args.get(key) is not None:
        hparams.set_key(key, args.get(key))


def _maybe_load_dataset_hparams(dataset_type, args, hparams):
    custom_root = os.path.join(__root__, "config", "data")
    dataset_type = dataset_type.lower()

    if hparams.get(dataset_type) is not None:
        name_from_args = args.get(f"{dataset_type}_name")
        # if not load
        if "root" not in hparams[dataset_type]:
            if name_from_args is not None:
                hparams[dataset_type].set_key("name", name_from_args)
                saber.log.info(f"set {dataset_type}.name = {name_from_args}")

            filename = saber_fs.maybe_in_dirs(
                hparams[dataset_type].name,
                must_be_found=True,
                possible_exts=[".json", ".py"],
                possible_roots=[custom_root],
            )
            hparams.overwrite_by(filename)
        elif name_from_args is not None:
            assert name_from_args == hparams[dataset_type]["name"], (
                f"args.{dataset_type}_name ({name_from_args}) != " +
                f"hparams.{dataset_type}.name ({hparams[dataset_type].name})"
            )

        # replace root
        _varn = "{" + f"{dataset_type.upper()}_ROOT" + "}"
        _root = saber_fs.maybe_remove_end_separator(hparams[dataset_type].root)
        saber.log.info(f"hparams: replace {_varn} into '{_root}'")
        hparams.replace_variable(_varn, _root)
    return hparams
