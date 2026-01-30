#!/bin/bash
#SBATCH --job-name=sail_pr_discdual_hc500e4
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_prdiscdual_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_prdiscdual_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== PR-SAIL: uniform sampling + weighted discriminator + dual-head (aux regularizer) =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# ---------------- Paths -----------------------------------------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# Teacher dataset (legacy exact)
EPISODES=4
ORIG_DATASET="${REPO_ROOT}/stable-baselines/teacher_dataset/legacy/expert_data_no_img_HalfCheetah_scores_500_episodes_8_e002_legacy_exact.npz"

TARGET_LINK="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_${EPISODES}.npz"

# Offline preference reward model (for PR-SAIL only)
PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

ENV_ID=HalfCheetah-v2
STEPS=1000000
SEEDS=(1)
# ---------------------------------------------------------------------

SID=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$SID]}

echo "Seed=${SEED}"
echo "Teacher dataset: ${ORIG_DATASET}"
echo "Pref RM: ${PREF_RM}"

# ---------------- Headless MuJoCo -------------------------------------
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

# ---------------- Environment -----------------------------------------
source "${VENV}/bin/activate"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"

cd "${REPO_ROOT}"
mkdir -p teacher_dataset

if [[ ! -f "${ORIG_DATASET}" ]]; then
  echo "ERROR: Teacher dataset missing: ${ORIG_DATASET}" >&2
  exit 2
fi

ln -sfn "${ORIG_DATASET}" "${TARGET_LINK}"

echo "Teacher dataset symlink:"
ls -l "${TARGET_LINK}"

cd "${RUN_DIR}"

COMMON_ARGS=(
  --env "${ENV_ID}"
  --algo sail
  --task gail-lfd-adaptive-dynamic
  --seed "${SEED}"
  --n-timesteps "${STEPS}"
  --n-episodes "${EPISODES}"
)

RUN_NAME="prdiscdual_hc1500_${STEPS}_s${SEED}"

# ---------------- W&B --------------------------------------------------
export WANDB_PROJECT="SAIL_variants"
export WANDB_NAME="${RUN_NAME}"
# ---------------------------------------------------------------------

if [[ ! -f "${PREF_RM}" ]]; then
  echo "Missing pref RM: ${PREF_RM}" >&2
  exit 3
fi

set -x
PYTHONPATH="$PWD/.." python -u train_sail.py \
  "${COMMON_ARGS[@]}" \
  --pref-reweight-teacher \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 17 \
  --pref-beta 5.0 \
  --pref-promote-quantile 0.5 \
  --pref-max-teacher-trajs 500 \
  \
  --disc-use-expert-weights \
  --disc-dualhead \
  --disc-aux-weight 1.0 \
  \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "Done: ${RUN_NAME}"
date