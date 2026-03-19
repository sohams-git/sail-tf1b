#!/bin/bash
#SBATCH --job-name=sail_tanh_tac_hc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_tanh_tac_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_tanh_tac_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: TANH Soft-TAC preference loss =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# ---------------- Paths ----------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

ENV_ID=HalfCheetah-v2
STEPS=1000000
EPISODES=4
SEEDS=(1)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Offline preference RM
PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

# Teacher dataset
ORIG_DATASET="${REPO_ROOT}/stable-baselines/teacher_dataset/legacy/expert_data_no_img_HalfCheetah_scores_500_episodes_${EPISODES}_legacy_exact.npz"
TARGET_LINK="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_${EPISODES}.npz"

# ------------- Env (CPU MuJoCo) -------------
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

# ------------- Activate env -------------
source "${VENV}/bin/activate"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"

# ------------- Dataset link -------------
cd "${REPO_ROOT}"
mkdir -p teacher_dataset
ln -sfn "${ORIG_DATASET}" "${TARGET_LINK}"

cd "${RUN_DIR}"

# ---------------- Fresh run dirs ----------------
RUN_NAME="SAIL_tanhSoftTAC_${STEPS}_s${SEED}"
RUN_LOG_DIR="logs/${RUN_NAME}"
RUN_TB_DIR="tb_logs/${RUN_NAME}"

rm -rf "${RUN_LOG_DIR}" "${RUN_TB_DIR}"

export WANDB_PROJECT="SAIL_variants"
export WANDB_NAME="${RUN_NAME}"
export WANDB_SILENT=true

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
  --pref-rank-weight 0.0 \
  --pref-rank-batch-size 32 \
  \
  --pref-soft-rank-weight 0.5 \
  --pref-soft-rank-temp 1.0 \
  --pref-tac-tie-eps 0.0 \
  \
  --pref-pair-filter none \
  --pref-pair-filter-tries 5000 \
  \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 17 \
  --pref-beta 5.0 \
  --pref-promote-quantile 0.5 \
  --pref-max-teacher-trajs 500 \
  \
  --tensorboard-log "${RUN_TB_DIR}" \
  --log-dir "${RUN_LOG_DIR}"
set +x

echo "Done: ${RUN_NAME}"
date