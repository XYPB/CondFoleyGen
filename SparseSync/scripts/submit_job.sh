#!/bin/bash

#### This scripts allows you to backup the working directory and submit the job from _that backed up state_
#### i.e any changes made in the working directory after you submitted a job will not affect the submitted script.
#### Submit jobs as (2 nodes with 4 GPUs in each == 8 GPUs in parallel for 15 mins with 500GB of local disk):
# ./submit_job.sh --partition=gputest --time=00:15:00 --nodes=2 --ntasks-per-node=4 --nvme=500 --file=./script.sh
#### where `script.sh`` is the script you would submit with the `sbatch ./script.sh` cmd.
#### If you cluster does not have local storage, drop the `--nvme=500` argument.


#### IMPORTANT: DEFINE YOUR USER VARIABLES HERE ##############################################################
# point this to the location of the working directory that you wish to backup once the sbatch job is submitted
ORIG_DIR="/projappl/project/SparseSync"
# where is the log dir (now points to the ORIG_DIR/logs)
LOG_DIR="$ORIG_DIR/logs/sync_models"
# GPU_TYPE is used in "--gres=gpu:$GPU_TYPE:X,nvme:XXXXX"
GPU_TYPE="v100"
# --mem-per-gpu=$MEM_PER_GPU
MEM_PER_GPU="85G"
# --cpus-per-task=$CPUS_PER_TASK
CPUS_PER_TASK="10"
# exclude from backup (file paths that have any of these pattern will be excluded from the backup copy)
EXCLUDE_PATTERNS=("logs" ".git" "data" "__pycache__" "*.pt" "*.pth" "sbatch_logs" "*.mp4" "*.wav" "*.jpg")
##############################################################################################################

# exit when any command fails
set -e

# argparse
for i in "$@"; do
  case $i in
    --partition=*)
      partition="${i#*=}"
      shift # past argument=value
      ;;
    --time=*)
      time="${i#*=}"
      shift # past argument=value
      ;;
    --nodes=*)
      NODES="${i#*=}"
      shift # past argument=value
      ;;
    --nvme=*)
      NVME="${i#*=}"
      shift # past argument=value
      ;;
    --ntasks-per-node=*)
      NTASKS_PER_NODE="${i#*=}"
      shift # past argument=value
      ;;
    --file=*)
      FILE="${i#*=}"
      shift # past argument=value
      ;;
    -* | --*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

echo "NVME:" $NVME
cd "$ORIG_DIR"

# timestamp + random second shift -- this will be used as the name of the folder with the experiment
NOW=$(python -c 'from datetime import timedelta, datetime; from random import randint; \
                 print((datetime.now() - timedelta(seconds=randint(0, 60))).strftime("%y-%m-%dT%H-%M-%S"))')

# the folder with experiment
LOG_DIR="$LOG_DIR/$NOW/code-$NOW"
mkdir -p "$LOG_DIR"

# backing up the code there
echo "Backing up the code to $LOG_DIR"
rsync -ah "${EXCLUDE_PATTERNS[@]/#/--exclude=}" . $LOG_DIR

# large files will be linked (not copied) to the original code (to save space, they are version controlled anyway)
ln -s $ORIG_DIR/data $LOG_DIR/data
ln -s $ORIG_DIR/logs $LOG_DIR/logs
ln -s $ORIG_DIR/model/modules/feat_extractors/visual/dino_deitsmall16_pretrain.pth $LOG_DIR/model/modules/feat_extractors/visual/dino_deitsmall16_pretrain.pth
ln -s $ORIG_DIR/model/modules/feat_extractors/visual/dino_deitsmall8_pretrain.pth $LOG_DIR/model/modules/feat_extractors/visual/dino_deitsmall8_pretrain.pth
ln -s $ORIG_DIR/model/modules/feat_extractors/visual/S3D_kinetics400_torchified.pt $LOG_DIR/model/modules/feat_extractors/visual/S3D_kinetics400_torchified.pt

# making dir for sbatch logs
mkdir $LOG_DIR/sbatch_logs

# selecting resources
GRES="gpu:$GPU_TYPE:$NTASKS_PER_NODE,nvme:$NVME"

# 15-33-68 <-- 22-03-14T15-33-68 -- splitting by 'T'
JOB_NAME=${NOW#*"T"}

echo "[$NOW]"
# submitting the job from within the experiment folder to avoid using the code state with unwanted changes
# also passing $NOW, so, when the job starts, if will know where to save stuff
sbatch \
    --job-name=$JOB_NAME \
    --partition=$partition \
    --time=$time \
    --mem-per-gpu=$MEM_PER_GPU \
    --gres=$GRES \
    --cpus-per-task=$CPUS_PER_TASK \
    --nodes=$NODES \
    --ntasks-per-node=$NTASKS_PER_NODE \
    --chdir=$LOG_DIR \
    --output=$ORIG_DIR/sbatch_logs/%J.log \
    --error=$ORIG_DIR/sbatch_logs/%J.log \
        $FILE --now=$NOW

#     --dependency=afterany:12517316 \
