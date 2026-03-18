#!/bin/bash
#SBATCH --job-name=sail_pref_rank_only_hc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_pref_rank_only_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_pref_rank_only_%A_%a.err
#SBATCH --account=sabymath0

# set -euo pipefail
# echo "== SAIL: Preference Ranking ONLY (no teacher reweight) =="
# date
# hostname
# echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
# RUN_DIR=${REPO_ROOT}/stable-baselines/run
# VENV=$HOME/.venvs/sail-tf1b

# ENV_ID=HalfCheetah-v2
# STEPS=1000000
# EPISODES=4
# SEEDS=(2 3 4 5)
# SID=$SLURM_ARRAY_TASK_ID
# SEED=${SEEDS[$SID]}

# PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

# # CPU MuJoCo
# export CUDA_VISIBLE_DEVICES=""
# export MUJOCO_GL=osmesa
# export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
# export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
# export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# export TF_CPP_MIN_LOG_LEVEL=2

# source "${VENV}/bin/activate"
# mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"
# cd "${RUN_DIR}"

# RUN_NAME="SAIL_Preference_Ranking_${STEPS}_s${SEED}"

# export WANDB_PROJECT="SAIL_variants"
# export WANDB_NAME="${RUN_NAME}"

# set -x
# PYTHONPATH="$PWD/.." python -u train_sail.py \
#   --env "${ENV_ID}" \
#   --algo sail \
#   --task gail-lfd-adaptive-dynamic \
#   --seed "${SEED}" \
#   --n-timesteps "${STEPS}" \
#   --n-episodes "${EPISODES}" \
#   \
#   --pref-rank-disc \
#   --pref-rank-weight 0.1 \
#   --pref-rank-batch-size 16 \
#   \
#   --pref-rm "${PREF_RM}" \
#   --pref-expect-obs-dim 17 \
#   \
#   --tensorboard-log "tb_logs/${RUN_NAME}" \
#   --log-dir        "logs/${RUN_NAME}"
# set +x

# echo "Done: ${RUN_NAME}"
# date





set -euo pipefail
echo "== SAIL: Preference Ranking ONLY (no teacher reweight) =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# ---- SAME teacher dataset as sbatch 1 -------------------
EPISODES=4
ORIG_DATASET="${REPO_ROOT}/stable-baselines/teacher_dataset/legacy/expert_data_no_img_HalfCheetah_scores_300_episodes_4_legacy_exact.npz"

TARGET_LINK="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_${EPISODES}.npz"

# ---------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------
ENV_ID=HalfCheetah-v2
STEPS=1000000
SEEDS=(2 3 4 5)

SID=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$SID]}

PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

echo "Seed=${SEED}"
echo "Original dataset: ${ORIG_DATASET}"
echo "Pref RM: ${PREF_RM}"

# ---------------------------------------------------------
# CPU / headless MuJoCo
# ---------------------------------------------------------
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export DISPLAY=
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

# ---------------------------------------------------------
# Activate env
# ---------------------------------------------------------
source "${VENV}/bin/activate"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"

cd "${RUN_DIR}"

# ---------------------------------------------------------
# Hyperparams safety check
# ---------------------------------------------------------
if [[ ! -f ../hyperparams/sail.yml ]]; then
  echo "[ERROR] ../hyperparams/sail.yml missing" >&2
  exit 1
fi

echo "[INFO] Using hyperparams file: $(readlink -f ../hyperparams/sail.yml)"

# ---------------------------------------------------------
# Teacher dataset symlink (same as sbatch 1)
# ---------------------------------------------------------
cd "${REPO_ROOT}"
mkdir -p teacher_dataset

if [[ ! -f "${ORIG_DATASET}" ]]; then
  echo "ERROR: Original dataset not found: ${ORIG_DATASET}" >&2
  exit 2
fi

ln -sfn "${ORIG_DATASET}" "${TARGET_LINK}"

echo "Teacher dataset symlink:"
ls -l "${TARGET_LINK}" || true

cd "${RUN_DIR}"

# ---------------------------------------------------------
# Run name
# ---------------------------------------------------------
RUN_NAME="SAIL_Preference_Ranking300_${STEPS}_s${SEED}"

export WANDB_PROJECT="SAIL_variants_HC"
export WANDB_NAME="${RUN_NAME}"

# ---------------------------------------------------------
# Safety check reward model
# ---------------------------------------------------------
if [[ ! -f "${PREF_RM}" ]]; then
  echo "Missing pref reward model: ${PREF_RM}" >&2
  exit 3
fi

# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
set -x
PYTHONPATH="$PWD/.." python -u train_sail.py \
  --env "${ENV_ID}" \
  --algo sail \
  --task gail-lfd-adaptive-dynamic \
  --seed "${SEED}" \
  --n-timesteps "${STEPS}" \
  --n-episodes "${EPISODES}" \
  \
  --pref-rank-disc \
  --pref-rank-weight 0.1 \
  --pref-rank-batch-size 16 \
  \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 17 \
  \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "Done: ${RUN_NAME}"
date