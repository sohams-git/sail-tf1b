#!/bin/bash
#SBATCH --job-name=sail_prefrew_hc500e8
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0                    # 1 run for now (index 0 only)
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_prefrew_500e8_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_prefrew_500e8_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL Pref-Reweighted (prefrew) CPU run on 500-score, 8-episode dataset =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# -------- Paths you control -------------------------------------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# Legacy teacher dataset you want to use (real file)
EPISODES=8
ORIG_DATASET="${REPO_ROOT}/stable-baselines/teacher_dataset/legacy/expert_data_no_img_HalfCheetah_scores_500_episodes_8_e002_legacy_exact.npz"

# IMPORTANT: this must match what train_sail.py expects:
# ../../teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_8.npz
TARGET_LINK="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_${EPISODES}.npz"

# Offline preference reward model (BPref / PEBBLE-style)
PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

ENV_ID=HalfCheetah-v2
STEPS=500000      # change for shorter/longer training
SEEDS=(1)          # 1 run for now
# ----------------------------------------------------------------------

SID=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$SID]}

echo "Seed=${SEED}"
echo "Original dataset: ${ORIG_DATASET}"

# ---- CPU / headless MuJoCo -------------------------------------------
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

# ---- Env + paths ------------------------------------------------------
source "${VENV}/bin/activate"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"

cd "${RUN_DIR}"
# Make hyperparams visible as ../hyperparams/sail.yml
test -f ../hyperparams/sail.yml || ln -sf td3.yml ../hyperparams/sail.yml

# Make sure teacher_dataset dir exists at REPO_ROOT and symlink expected name -> legacy file
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

COMMON_ARGS=(
  --env "${ENV_ID}"
  --algo sail
  --task gail-lfd-adaptive-dynamic
  --seed "${SEED}"
  --n-timesteps "${STEPS}"
  --n-episodes "${EPISODES}"
)

RUN_NAME="prefrew_hrunt1_hc500e8_v2_${STEPS}_s${SEED}"

# Safety check for pref model
if [[ ! -f "${PREF_RM}" ]]; then
  echo "Missing pref reward model: ${PREF_RM}" >&2
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
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "Done: ${RUN_NAME}"
date