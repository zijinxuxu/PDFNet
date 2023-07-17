#!/usr/bin/env bash
set -ex

# Training
GPU_ID=0
NAME='demo'
task='interact' # simplified, artificial
mode='test'
dataset='H2O' 
# Network configuration

BATCH_SIZE=1

# Reconstruction resolution
# NOTE: one can change here to reconstruct mesh in a different resolution.
Input_RES=384 # 224,384

CHECKPOINTS_PATH='data/Best_H2O_ours_PDF_center_2gpu_56.pth'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port 12507 --nproc_per_node 1 demo.py \
    --task ${task} \
    --gpus 0 \
    --mode ${mode} \
    --dataset ${dataset} \
    --batch_size ${BATCH_SIZE} \
    --default_resolution ${Input_RES} \
    --bone_loss \
    --arch csp_50 \
    --avg_center \
    --reproj_loss \
    --config_info H2O3D-PDF-ctfeat-noimgfaet-2gpu-384-aug-repeat50\
    --depth \
    --lr 1e-5 \
    --load_model ${CHECKPOINTS_PATH}