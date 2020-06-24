import sys
import saber
import argparse
import importlib
from .api import train_model, evaluate_model


if __name__ == "__main__":
    choices = ["train", "evaluate", "merge_videos"]

    parser = argparse.ArgumentParser()
    parser.add_argument("mode",             type=str, help="how to run this module", choices=choices)
    # global settings
    parser.add_argument("--tag",            type=str, help="overwrite 'tag'")
    parser.add_argument("--matplotlib_use", type=str, help="matploblib.use()", default="Agg")
    parser.add_argument("--log_dir",        type=str, help="overwrite 'log_dir'")
    parser.add_argument("--load_from",      type=str, help="load pre-trained model")
    parser.add_argument("--custom_hparams", type=str, help="overwrite hparams from json")
    parser.add_argument("--ensembling_ms",  type=int, help="overwrite 'ensembling_ms'")
    parser.add_argument("--save_video",     action="store_true")
    # merge videos
    parser.add_argument("--output_path",    type=str)
    parser.add_argument("--video_list",     nargs="+")
    parser.add_argument("--fps",            type=int, default=60)
    parser.add_argument("--sr",             type=int, default=16000)
    parser.add_argument("--grid_w",         type=int, default=780)
    parser.add_argument("--grid_h",         type=int, default=780)
    parser.add_argument("--rows",           type=int, default=0)
    parser.add_argument("--cols",           type=int, default=0)
    parser.add_argument("--font_size",      type=int, default=40)
    parser.add_argument("--indices",        nargs="*")
    parser.add_argument("--delay_list",     nargs="*")
    parser.add_argument("--title_list",     nargs="*")
    parser.add_argument("--crop_list",      nargs="*")
    parser.add_argument("--overwrite",      action="store_true")

    args = saber.ConfigDict(parser.parse_args())

    if args.mode == "train":
        train_model(args)
    elif args.mode == "evaluate":
        evaluate_model(args)
    else:
        raise NotImplementedError()
