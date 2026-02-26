#!/bin/bash
#SBATCH --job-name=sail_vanilla_hopper
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_hopper_vanilla_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_hopper_vanilla_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail

echo "======================================="
echo "Vanilla SAIL Full Run (ARRAY) - Hopper-v2"
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"
echo "======================================="

# --------------------------------------------------
# Paths
# --------------------------------------------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# --------------------------------------------------
# Environment settings
# --------------------------------------------------
ENV_ID=Hopper-v2
STEPS=1000000
EPISODES=10
VAR=vanilla

# Define seeds
SEEDS=(2 3 4)
SID=$SLURM_ARRAY_TASK_ID

# Guard against mismatch
if [[ ${SID} -ge ${#SEEDS[@]} ]]; then
  echo "ERROR: array task id ${SID} out of range for SEEDS (len=${#SEEDS[@]})" >&2
  exit 4
fi

SEED=${SEEDS[$SID]}
RUN_NAME="vanillaSAIL_hopper_${STEPS}_s${SEED}"

echo "Seed=${SEED}"

# --------------------------------------------------
# Dataset
# --------------------------------------------------
FORCE_DATASET=${REPO_ROOT}/teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_${EPISODES}.npz

# --------------------------------------------------
# Logging
# --------------------------------------------------
LOG_DIR=${RUN_DIR}/logs/${RUN_NAME}
TB_DIR=${RUN_DIR}/tb_logs/${RUN_NAME}

rm -rf "${LOG_DIR}"
rm -rf "${TB_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${TB_DIR}"

# --------------------------------------------------
# MuJoCo
# --------------------------------------------------
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1

# --------------------------------------------------
# W&B
# --------------------------------------------------
export WANDB_PROJECT="SAIL_variants_hopper"
export WANDB_NAME="${RUN_NAME}"
export WANDB_GROUP="${ENV_ID}_${VAR}_${STEPS}"

echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "WANDB_GROUP=${WANDB_GROUP}"

# --------------------------------------------------
# Activate environment
# --------------------------------------------------
source "${VENV}/bin/activate"
cd "${RUN_DIR}"

echo "Using python: $(which python)"
python --version

# --------------------------------------------------
# Dataset symlink (⚠ race warning for arrays)
# --------------------------------------------------
echo "Forcing dataset: ${FORCE_DATASET}"
rm -f ../teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_${EPISODES}.npz
ln -s "${FORCE_DATASET}" ../teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_${EPISODES}.npz

echo "Resolved dataset:"
readlink -f ../teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_${EPISODES}.npz

# --------------------------------------------------
# Run Vanilla SAIL
# --------------------------------------------------
set -x
PYTHONPATH="$PWD/.." python -u train_sail.py \
  --env "${ENV_ID}" \
  --seed "${SEED}" \
  --algo sail \
  --task gail-lfd-adaptive-dynamic \
  --n-timesteps "${STEPS}" \
  --n-episodes "${EPISODES}" \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir "logs/${RUN_NAME}"
set +x

echo "======================================="
echo "Training Finished: ${RUN_NAME}"
date
echo "======================================="