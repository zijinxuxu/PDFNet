#!/usr/bin/env bash
set -ex

# Training
GPU_ID=2,3
NAME='SMHR_train'
task='interact' 
mode='train' # train, test, val
dataset='H2O' # modify datasets in data/joint_dataset.txt
# Network configuration

BATCH_SIZE=8

# Reconstruction resolution
Input_RES=384 # 224,512

CHECKPOINTS_PATH='/home/zijinxuxu/codes/SMHR-InterHand/outputs/model_dump/Best_H2O_ours_PDF_center_2gpu_56.pth'
# CHECKPOINTS_PATH='/home/zijinxuxu/codes/SMHR-InterHand/interhand.pth'

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --master_port 12507 --nproc_per_node 2 main.py \
    --task ${task} \
    --gpus 0,1 \
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
    # -withnormal-withFPS
    # --sample_strategy FPS \
    # --load_model ${CHECKPOINTS_PATH}
    # --depth \
    # --load_model ${CHECKPOINTS_PATH} 
    # --depth \
    # --load_model ${CHECKPOINTS_PATH} 
    # --joints_weight 0. \
    # --mano_weight 0. \
    # --load_model ${CHECKPOINTS_PATH} 
    # --joints_weight 0. \
    # --mano_weight 0. \
    # --load_model ${CHECKPOINTS_PATH} \
    # --mano_weight 0. \
    # --off \
    # --depth \
    # set to true when using pca rather than euler angles
    # --using_pca \ 
    # set to true when using heatmaps of keypoints
    #--heatmaps \ 
    # for FreiHAND and HO3D dataset without detection
    #--no_det \ 
    # used when both left/right hand exists.
    # --pick_hand \
    # set to true when using bone_dir_loss. 
    #--bone_loss \
    # set to true when using pho_loss.
    # --photometric_loss \ 
