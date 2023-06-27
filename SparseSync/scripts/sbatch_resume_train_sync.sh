#!/bin/bash

#SBATCH --job-name=cts
#SBATCH --account=project_2000936
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1,nvme:500
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=85G

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

# load conda environment
source $PROJAPPL/miniconda3/etc/profile.d/conda.sh
conda activate sparse_sync

SYNC_LOGS_PATH=$SCRATCH/vladimir/logs/sync/sync_models/
CKPT_ID="xx-xx-xxTxx-xx-xx"  # replace this with exp folder name

srun python main.py \
    config="$SYNC_LOGS_PATH/$CKPT_ID/cfg-$CKPT_ID.yaml" \
    training.resume="True" training.finetune="False"
    # logging.log_code_state=False training.finetune="False" training.run_test_only="True" data.dataset.params.iter_times="25" data.dataset.params.load_fixed_offsets_on_test="False"
