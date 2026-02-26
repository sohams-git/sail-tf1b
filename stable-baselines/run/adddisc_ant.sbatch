#!/bin/bash
#SBATCH --job-name=sail_adddisc_ant
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_ant_adddisc_%j.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_ant_adddisc_%j.err
#SBATCH --account=sabymath0

set -euo pipefail

echo "======================================="
echo "SAIL add-pref-to-disc TEST - Ant-v2"
date
hostname
echo "JOB ID: $SLURM_JOB_ID"
echo "======================================="

# ------------------------------
# Paths
# ------------------------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# ------------------------------
# Dataset (ANT)
# ------------------------------
FORCE_DATASET=${REPO_ROOT}/teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_10.npz
EPISODES=10

# ------------------------------
# Reward Model (ANT)
# ------------------------------
PREF_RM=/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/ant_pebble_1M/reward_model_1000000_0.pt

# ------------------------------
# Run naming
# ------------------------------
ENV_ID=Ant-v2
STEPS=1000000
SEED=1
VAR=adddisc
RUN_NAME="discsail1200_${STEPS}_s${SEED}"

LOG_DIR=${RUN_DIR}/logs/${RUN_NAME}
mkdir -p "${LOG_DIR}"

# ------------------------------
# MuJoCo
# ------------------------------
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1

# ------------------------------
# WandB (Ant Project)
# ------------------------------
export WANDB_PROJECT="SAIL_variants_ant"
export WANDB_NAME="${RUN_NAME}"
export WANDB_GROUP="${ENV_ID}_${VAR}_${STEPS}"
# export WANDB_ENTITY="sohams1-university-of-michigan"  # if needed

echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"

# ------------------------------
# Activate environment
# ------------------------------
source "${VENV}/bin/activate"
cd "${RUN_DIR}"

echo "Using python: $(which python)"
python --version

# ------------------------------
# Safety check
# ------------------------------
if [[ ! -f "${PREF_RM}" ]]; then
  echo "[ERROR] Missing pref reward model: ${PREF_RM}" >&2
  exit 3
fi

if [[ ! -f "${FORCE_DATASET}" ]]; then
  echo "[ERROR] Missing teacher dataset: ${FORCE_DATASET}" >&2
  exit 4
fi

# ------------------------------
# Force dataset symlink (ANT)
# ------------------------------
echo "Forcing dataset: ${FORCE_DATASET}"
rm -f "../teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_${EPISODES}.npz"
ln -s "${FORCE_DATASET}" "../teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_${EPISODES}.npz"

echo "Resolved dataset:"
readlink -f "../teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_${EPISODES}.npz"

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
  --add-pref-to-disc \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 111 \
  --pref-weight 2.0 \
  --pref-norm zscore \
  --pref-clip 5.0 \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir "logs/${RUN_NAME}"
set +x

echo "======================================="
echo "Training Finished: ${RUN_NAME}"
date
echo "======================================="