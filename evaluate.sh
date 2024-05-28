#!/bin/bash

# ---------------------------------------------------------------------------- #
# - Generate animation in FLAME topology.
# ---------------------------------------------------------------------------- #

# Load from pretrained checkpoint.
LOAD_CKPT="./pretrained_models/dgrad/checkpoints/epoch0050-step086751.ckpt"
LOAD_HPARAMS="./pretrained_models/dgrad/hparams.json"
# Input data.
EVAL_OUT_DIR="./results_flame/"
EVAL_INPUT="~/assets/eval_data/clips/speech@clip0.mp4"  # Change to your audio input.
EVAL_SPK_COND="m1"  # Which speaker style is used as condition.

python3 -m speech_anime evaluate \
    --load_from         "$LOAD_CKPT"        \
    --custom_hparams    "$LOAD_HPARAMS"     \
    --output_dir        "$EVAL_OUT_DIR"     \
    --eval_input        "$EVAL_INPUT"       \
    --eval_spk_cond     "$EVAL_SPK_COND"    \
    --overwrite_video                       \
;

# ---------------------------------------------------------------------------- #
# - Different topology with mesh triangle correspondance.
# ---------------------------------------------------------------------------- #
# Replace the 'FW_TMPLT'
# CHECKPOINT="assets/pretrained_model/epoch0051-step089046.ckpt"
# HPARAMS="assets/pretrained_model/hparams.json"
# FW_TMPLT="assets/facewarehouse_template.obj"
# FW_RESULTS_DIR="./results_fw/"

# python3 -m speech_anime evaluate \
#     --load_from         "$CHECKPOINT"                         \
#     --custom_hparams    "$HPARAMS"                            \
#     --template_mesh     "$FW_TMPLT"                           \
#     --mesh_constraints  assets/fw_indices/constraints.txt     \
#     --mesh_tricorres    assets/fw_indices/out.tricorrs        \
#     --output_dir        "$FW_RESULTS_DIR"                     \
#     --overwrite_video                                         \
#     --export_mesh_frames                                  ;
