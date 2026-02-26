#!/bin/bash
#SBATCH --job-name=sail_softtac_hc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_softtac_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_softtac_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: Soft-TAC ONLY (no reweighting, no hard ranking) =="
date
hostname
echo "JOB=$SLURM_JOB_ID"

REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

ENV_ID=HalfCheetah-v2
STEPS=1000000
EPISODES=4
SEEDS=(6)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Offline preference RM (BPref export)
PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

# Teacher dataset link MUST live in stable-baselines/teacher_dataset for ../teacher_dataset resolution
ORIG_DATASET="${REPO_ROOT}/stable-baselines/teacher_dataset/legacy/expert_data_no_img_HalfCheetah_scores_500_episodes_${EPISODES}_legacy_exact.npz"
TEACHER_DIR="${REPO_ROOT}/stable-baselines/teacher_dataset"
TARGET_LINK="${TEACHER_DIR}/expert_data_no_img_HalfCheetah_scores_5600_episodes_${EPISODES}.npz"

# ------------- CPU MuJoCo -------------
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

source "${VENV}/bin/activate"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"

# ------------- Dataset link -------------
mkdir -p "${TEACHER_DIR}"
ln -sfn "${ORIG_DATASET}" "${TARGET_LINK}"

cd "${RUN_DIR}"

RUN_NAME="SAIL_softTAC_only_${STEPS}_s${SEED}"

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
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 17 \
  \
  --pref-rank-disc \
  --pref-rank-weight 0.0 \
  --pref-rank-batch-size 32 \
  \
  --pref-soft-rank-weight 0.5 \
  --pref-soft-rank-temp 1.0 \
  \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "Done: ${RUN_NAME}"
date