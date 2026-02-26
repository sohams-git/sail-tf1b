#!/bin/bash
#SBATCH --job-name=sail_adddisc_prefrew_hopper
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_hopper_adddisc_prefrew_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_hopper_adddisc_prefrew_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail

echo "======================================="
echo "SAIL add-pref-to-disc + pref-reweight-teacher (ARRAY) - Hopper-v2"
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"
echo "======================================="

# ------------------------------
# Paths
# ------------------------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# ------------------------------
# Dataset
# ------------------------------
FORCE_DATASET=${REPO_ROOT}/teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_10.npz
EPISODES=10

# ------------------------------
# Reward Model
# ------------------------------
PREF_RM=/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/hopper_pebble_1M/reward_model_1000000_0.pt

# ------------------------------
# Reweight hyperparams (same as prefrew sbatch)
# ------------------------------
PREF_BETA=1.0
PREF_PROMOTE_QUANTILE=0.5
PREF_MAX_TEACHER_TRAJS=500

# ------------------------------
# Run naming / seeds
# ------------------------------
ENV_ID=Hopper-v2
STEPS=1000000
VAR=adddisc_prefrew

SEEDS=(0)
SID=$SLURM_ARRAY_TASK_ID

# Guard: prevent out-of-range if someone changes --array but not SEEDS
if [[ ${SID} -ge ${#SEEDS[@]} ]]; then
  echo "ERROR: array task id ${SID} out of range for SEEDS (len=${#SEEDS[@]})" >&2
  exit 4
fi

SEED=${SEEDS[$SID]}

RUN_NAME="PAIL_rewshaping_hopper__${STEPS}_s${SEED}"

echo "Seed=${SEED}"
echo "Dataset: ${FORCE_DATASET}"
echo "Pref RM: ${PREF_RM}"

LOG_DIR=${RUN_DIR}/logs/${RUN_NAME}
mkdir -p "${LOG_DIR}"

# ------------------------------
# MuJoCo
# ------------------------------
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1

# ------------------------------
# WandB (Hopper Project)
# ------------------------------
export WANDB_PROJECT="SAIL_variants_hopper"
export WANDB_NAME="${RUN_NAME}"
export WANDB_GROUP="${ENV_ID}_${VAR}_${STEPS}"

echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "WANDB_GROUP=${WANDB_GROUP}"

# ------------------------------
# Activate environment
# ------------------------------
source "${VENV}/bin/activate"
cd "${RUN_DIR}"

echo "Using python: $(which python)"
python --version

# ------------------------------
# Safety checks
# ------------------------------
if [[ ! -f "${FORCE_DATASET}" ]]; then
  echo "ERROR: Dataset not found: ${FORCE_DATASET}" >&2
  exit 2
fi

if [[ ! -f "${PREF_RM}" ]]; then
  echo "ERROR: Pref reward model not found: ${PREF_RM}" >&2
  exit 3
fi

# ------------------------------
# Force dataset symlink
# NOTE: this is safe for arrays only if each task runs in isolated repo dir.
# If all array tasks share the same repo path simultaneously, they will race.
# ------------------------------
echo "Forcing dataset: ${FORCE_DATASET}"
rm -f "../teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_${EPISODES}.npz"
ln -s "${FORCE_DATASET}" "../teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_${EPISODES}.npz"

echo "Resolved dataset:"
readlink -f "../teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_${EPISODES}.npz"

# ------------------------------
# Run training
# ------------------------------
set -x
PYTHONPATH="$PWD/.." python -u train_sail.py \
  --env "${ENV_ID}" \
  --seed "${SEED}" \
  --algo sail \
  --task gail-lfd-adaptive-dynamic \
  --n-timesteps "${STEPS}" \
  --n-episodes "${EPISODES}" \
  \
  --add-pref-to-disc \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 11 \
  --pref-weight 0.5 \
  --pref-norm zscore \
  --pref-clip 5.0 \
  \
  --pref-reweight-teacher \
  --pref-beta "${PREF_BETA}" \
  --pref-promote-quantile "${PREF_PROMOTE_QUANTILE}" \
  --pref-max-teacher-trajs "${PREF_MAX_TEACHER_TRAJS}" \
  \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir "logs/${RUN_NAME}"
set +x

echo "======================================="
echo "Training Finished: ${RUN_NAME}"
date
echo "======================================="