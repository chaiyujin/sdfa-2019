import sys
import saber
import argparse
import importlib
from .api import train_model, evaluate_model, jit_trace


if __name__ == "__main__":
    choices = ["train", "evaluate", "trace"]

    parser = argparse.ArgumentParser()
    parser.add_argument("mode",               type=str, help="how to run this module", choices=choices)
    # global settings
    parser.add_argument("--tag",              type=str, help="overwrite 'tag'")
    parser.add_argument("--matplotlib_use",   type=str, help="matploblib.use()", default="Agg")
    parser.add_argument("--log_dir",          type=str, help="overwrite 'log_dir'")
    parser.add_argument("--load_from",        type=str, help="load pre-trained model")
    parser.add_argument("--custom_hparams",   type=str, help="overwrite hparams from json")
    parser.add_argument("--ensembling_ms",    type=int, help="overwrite 'ensembling_ms'")
    parser.add_argument("--save_video",       action="store_true")
    # generate videos
    parser.add_argument("--output_dir",       type=str)
    parser.add_argument("--grid_w",           type=int, default=512)
    parser.add_argument("--grid_h",           type=int, default=512)
    parser.add_argument("--font_size",        type=int, default=24)
    parser.add_argument("--overwrite_video",  action="store_true")
    parser.add_argument("--with_title",       action="store_true")
    parser.add_argument("--draw_truth",       action="store_true")
    parser.add_argument("--draw_align",       action="store_true")
    parser.add_argument("--draw_latent",      action="store_true")
    # dump traced
    parser.add_argument("--traced_dump_path", type=str, help="path to dump traced model")
    # parse args
    args = saber.ConfigDict(parser.parse_args())

    if args.mode == "train":
        train_model(args)
    elif args.mode == "evaluate":
        evaluate_model(args)
    elif args.mode == "trace":
        jit_trace(args)
    else:
        raise NotImplementedError()
