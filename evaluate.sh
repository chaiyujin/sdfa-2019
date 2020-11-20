#!/bin/bash
# ---------------------------------------------------------------------------- #
# 1. Set source audio (or video file)
#   by add records in 'hparams.trainer.evaluate.test' in 
#   file 'speech_anime/config/default.py'
# ---------------------------------------------------------------------------- #
# 2. Replace the 'FW_TMPLT'
CHECKPOINT="assets/pretrained_model/epoch0051-step089046.ckpt"
HPARAMS="assets/pretrained_model/hparams.json"
FW_TMPLT="assets/facewarehouse_template.obj"
FW_RESULTS_DIR="./results_fw/"

python3 -m speech_anime evaluate \
    --load_from         "$CHECKPOINT"                         \
    --custom_hparams    "$HPARAMS"                            \
    --template_mesh     "$FW_TMPLT"                           \
    --mesh_constraints  assets/fw_indices/constraints.txt     \
    --mesh_tricorres    assets/fw_indices/out.tricorrs        \
    --output_dir        "$FW_RESULTS_DIR"                     \
    --overwrite_video                                         \
    --export_mesh_frames                                  ;
# ---------------------------------------------------------------------------- #
# 3. following is the cmd to generate animation with original flame tempalte
FLAME_RESULTS_DIR="./results_flame/"

python3 -m speech_anime evaluate \
    --load_from         "$CHECKPOINT"               \
    --custom_hparams    "$HPARAMS"                  \
    --output_dir        "$FLAME_RESULTS_DIR"        \
    --overwrite_video                               \
;  #    --export_mesh_frames                            ;
# ---------------------------------------------------------------------------- #
