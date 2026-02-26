#!/bin/bash
#SBATCH --job-name=sail_vanilla_walker
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_walker_vanilla_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_walker_vanilla_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail

echo "======================================="
echo "Vanilla SAIL Full Run (ARRAY) - Walker2d-v2"
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
ENV_ID=Walker2d-v2
STEPS=1000000
EPISODES=10
VAR=vanilla

# Define seeds
SEEDS=(0 1)
SID=$SLURM_ARRAY_TASK_ID

if [[ ${SID} -ge ${#SEEDS[@]} ]]; then
  echo "ERROR: array task id ${SID} out of range for SEEDS (len=${#SEEDS[@]})" >&2
  exit 4
fi

SEED=${SEEDS[$SID]}
RUN_NAME="vanillaSAIL_walker_${STEPS}_s${SEED}"
echo "Seed=${SEED}"

# --------------------------------------------------
# Dataset (source)
# --------------------------------------------------
FORCE_DATASET=${REPO_ROOT}/teacher_dataset/expert_data_no_img_Walker2d_scores_1500_episodes_${EPISODES}.npz

# --------------------------------------------------
# Logging
# --------------------------------------------------
LOG_DIR=${RUN_DIR}/logs/${RUN_NAME}
TB_DIR=${RUN_DIR}/tb_logs/${RUN_NAME}

rm -rf "${LOG_DIR}" "${TB_DIR}"
mkdir -p "${LOG_DIR}" "${TB_DIR}"

# --------------------------------------------------
# MuJoCo
# --------------------------------------------------
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1

# --------------------------------------------------
# W&B
# --------------------------------------------------
export WANDB_PROJECT="SAIL_variants_walker"
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
# Dataset wiring (array-safe)
# Make a per-task teacher_dataset dir and point SAIL to it
# --------------------------------------------------
TASK_TEACHER_DIR="${RUN_DIR}/.teacher_dataset_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TASK_TEACHER_DIR}"

TARGET_NAME="expert_data_no_img_Walker2d_scores_1500_episodes_${EPISODES}.npz"
ln -sf "${FORCE_DATASET}" "${TASK_TEACHER_DIR}/${TARGET_NAME}"

# SAIL expects ../teacher_dataset from RUN_DIR, so we temporarily repoint it
# Backup original symlink/dir name (if any) using a unique backup per task
BACKUP_PATH="../teacher_dataset._bak_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

if [[ -e ../teacher_dataset && ! -L ../teacher_dataset ]]; then
  echo "ERROR: ../teacher_dataset is a real directory (not symlink). Not modifying it for safety." >&2
  echo "Make ../teacher_dataset a symlink if you want array-safe swapping, or run single-seed jobs." >&2
  exit 7
fi

# Move existing symlink out of the way (if present), then point to task dir
if [[ -L ../teacher_dataset ]]; then
  mv ../teacher_dataset "${BACKUP_PATH}"
fi
ln -s "${TASK_TEACHER_DIR}" ../teacher_dataset

echo "Resolved dataset:"
readlink -f "../teacher_dataset/${TARGET_NAME}"

# Ensure we restore teacher_dataset symlink on exit
cleanup() {
  rm -f ../teacher_dataset
  if [[ -e "${BACKUP_PATH}" ]]; then
    mv "${BACKUP_PATH}" ../teacher_dataset
  fi
}
trap cleanup EXIT

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