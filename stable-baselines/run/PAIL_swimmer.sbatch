#!/bin/bash
#SBATCH --job-name=sail_pail_swimmer
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_swimmer_pail_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_swimmer_pail_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail

echo "======================================="
echo "PAIL-style (Pref-Reweight Teacher) - Swimmer-v2"
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
ENV_ID=Swimmer-v2
STEPS=1000000
EPISODES=4
VAR=prefrew   # naming for W&B grouping

# seed(s)
SEEDS=(5 6 7 8 9)
SID=$SLURM_ARRAY_TASK_ID
if [[ ${SID} -ge ${#SEEDS[@]} ]]; then
  echo "ERROR: array task id ${SID} out of range for SEEDS (len=${#SEEDS[@]})" >&2
  exit 4
fi
SEED=${SEEDS[$SID]}

# --------------------------------------------------
# Dataset (source + expected target path)
# IMPORTANT: match vanilla's "TARGET_DATASET" location
# --------------------------------------------------
FORCE_DATASET=${REPO_ROOT}/teacher_dataset/expert_data_no_img_Swimmer_scores_120_episodes_${EPISODES}.npz
TARGET_DATASET=${RUN_DIR}/../teacher_dataset/expert_data_no_img_Swimmer_scores_120_episodes_${EPISODES}.npz

# --------------------------------------------------
# Pref RM (Swimmer) + dims from your eval
# --------------------------------------------------
PREF_RM=/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/swimmer_pebble_1M/reward_model_1000000_0.pt
PREF_OBS_DIM=8  # from your eval output

# --------------------------------------------------
# Logging
# --------------------------------------------------
RUN_NAME="PAIL_swimmer120_beta0.5_${STEPS}_s${SEED}"
LOG_DIR=${RUN_DIR}/logs/${RUN_NAME}
TB_DIR=${RUN_DIR}/tb_logs/${RUN_NAME}

rm -rf "${LOG_DIR}" "${TB_DIR}"
mkdir -p "${LOG_DIR}" "${TB_DIR}"

# --------------------------------------------------
# MuJoCo (match vanilla for GL)
# --------------------------------------------------
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1

# --------------------------------------------------
# W&B (match vanilla)
# --------------------------------------------------
export WANDB_PROJECT="SAIL_variants_swimmer"
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
# Safety checks
# --------------------------------------------------
if [[ ! -f "${FORCE_DATASET}" ]]; then
  echo "ERROR: Dataset not found: ${FORCE_DATASET}" >&2
  exit 2
fi
if [[ ! -f "${PREF_RM}" ]]; then
  echo "ERROR: Pref reward model not found: ${PREF_RM}" >&2
  exit 3
fi

# --------------------------------------------------
# Ensure dataset exists where SAIL will look (array-safe)
# --------------------------------------------------
if [[ ! -f "${TARGET_DATASET}" ]]; then
  echo "Copying dataset into ../teacher_dataset/ ..."
  cp -f "${FORCE_DATASET}" "${TARGET_DATASET}"
fi

echo "Dataset in use:"
ls -lh "${TARGET_DATASET}"

# --------------------------------------------------
# Run PAIL-style (Pref-Reweight Teacher)
# --------------------------------------------------
set -x
PYTHONPATH="$PWD/.." python -u train_sail.py \
  --env "${ENV_ID}" \
  --seed "${SEED}" \
  --algo sail \
  --task gail-lfd-adaptive-dynamic \
  --n-timesteps "${STEPS}" \
  --n-episodes "${EPISODES}" \
  --pref-reweight-teacher \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim "${PREF_OBS_DIM}" \
  --pref-beta 1.0 \
  --pref-promote-quantile 0.5 \
  --pref-max-teacher-trajs 500 \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir "logs/${RUN_NAME}"
set +x

echo "======================================="
echo "Training Finished: ${RUN_NAME}"
date
echo "======================================="