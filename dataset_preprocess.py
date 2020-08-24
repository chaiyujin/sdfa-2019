import os
import argparse
import matplotlib
from speech_anime.datasets.vocaset.preload import (
    clean_voca, preload_voca, generate_dgrad,
    pca_offsets, pca_dgrad
)
matplotlib.use('Agg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=str,   required=True)
    parser.add_argument("--output_root", type=str,   required=True)
    parser.add_argument("--sample_rate", type=int,   default=8000)
    parser.add_argument("--target_db",   type=float, default=-24.5)
    parser.add_argument("--debug_audio", action="store_true")
    parser.add_argument("--debug_video", action="store_true")
    args = parser.parse_args()

    clean_voca(
        root=args.source_root,
        clean_root=os.path.join(args.output_root, "clean"),
        debug_root=os.path.join(args.output_root, "debug"),
        sample_rate=args.sample_rate,
        target_db=args.target_db
    )

    preload_voca(
        root        = args.source_root,
        clean_root  = os.path.join(args.output_root, "clean"),
        output_root = os.path.join(args.output_root, "offsets"),
        sample_rate = args.sample_rate,
        debug_audio = False,
        debug_video = False,
    )

    generate_dgrad(
        offsets_root = os.path.join(args.output_root, "offsets"),
        dgrad_root   = os.path.join(args.output_root, "dgrad")
    )

    # PCA for offsets and dgrad
    pca_offsets(os.path.join(args.output_root, "offsets"))
    pca_dgrad(os.path.join(args.output_root, "dgrad"))
