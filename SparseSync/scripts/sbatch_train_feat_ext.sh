#!/bin/bash

#SBATCH --job-name=tfe
#SBATCH --account=project_2000936
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1,nvme:500
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=30G

# exit when any command fails
set -e

# loading conda environment
source $PROJAPPL/miniconda3/etc/profile.d/conda.sh
conda activate sparse_sync

## The following will assign a master port (picked randomly to avoid collision) and an address for ddp.
## However, you will need just one GPU
# We want names of master and slave nodes. Make sure this node (MASTER_ADDR) comes first
MASTER_ADDR=`/bin/hostname -s`
if (( $SLURM_JOB_NUM_NODES > 1 )); then
    WORKERS=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER_ADDR`
fi
# Get a random unused port on this host(MASTER_ADDR)
MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export MASTER_PORT=$MASTER_PORT
export MASTER_ADDR=$MASTER_ADDR

# path to the folder with `.wav` files
VIDS_PATH="/scratch/project_2000936/vladimir/vggsound/h264_video_25fps_256side_16000hz_aac/"

srun python main.py \
    config="./configs/audio_feature_extractor.yaml" \
    data.vids_dir="$VIDS_PATH" \
    data.crop_len_sec="5"
