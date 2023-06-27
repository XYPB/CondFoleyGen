#!/bin/bash

#SBATCH --job-name=ts
#SBATCH --account=project_2000936
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1,nvme:500
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=85G

# argparse
for i in "$@"; do
  case $i in
    -n=*|--now=*)
      NOW="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

# exit when any command fails
set -e

## The following will assign a master port (picked randomly to avoid collision) and an address for ddp.
# We want names of master and slave nodes. Make sure this node (MASTER_ADDR) comes first
MASTER_ADDR=`/bin/hostname -s`
if (( $SLURM_JOB_NUM_NODES > 1 )); then
    WORKERS=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER_ADDR`
fi
# Get a random unused port on this host(MASTER_ADDR)
MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export MASTER_PORT=$MASTER_PORT
export MASTER_ADDR=$MASTER_ADDR
echo "MASTER_ADDR" $MASTER_ADDR "MASTER_PORT" $MASTER_PORT "WORKERS" $WORKERS

# loading conda environment
source $PROJAPPL/miniconda3/etc/profile.d/conda.sh
conda activate sparse_sync

# init weights from a model pre-trained on another dataset -- use the experiment id corresponding to the pre-training run
CKPT_ID="22-07-13T22-25-49"

srun python main.py \
    start_time="$NOW" \
    config="./configs/sparse_sync.yaml" \
    training.finetune="True" \
    data.vids_path="$SCRATCH/vladimir/vggsound/h264_video_25fps_256side_16000hz_aac/" \
    logging.logdir="$SCRATCH/vladimir/logs/sync/sync_models/" \
    ckpt_path="$SCRATCH/vladimir/logs/sync/sync_models/$CKPT_ID/$CKPT_ID.pt" \
    data.dataset.target="dataset.vggsound.VGGSoundSparsePicked" \
    data.audio_jitter_sec="0.05" \
    data.p_horizontal_flip="0.5" \
    data.sometimes_upscale_p="0.5" \
    data.p_gray_scale="0.2" \
    data.p_color_jitter="0.2" \
    data.p_audio_aug="0.2" \
    training.base_batch_size="10"
