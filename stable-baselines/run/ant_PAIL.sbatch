#!/bin/bash
#SBATCH --job-name=sail_prefrew_ant
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_prefrew_ant_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_prefrew_ant_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL Pref-Reweighted (PR-SAIL) on Ant teacher dataset =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# ------------------------------
# Paths
# ------------------------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# ------------------------------
# Dataset (NO symlink needed)
# ------------------------------
FORCE_DATASET=${REPO_ROOT}/teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_10.npz
EPISODES=10

# ------------------------------
# Ant Reward Model
# ------------------------------
PREF_RM=/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/ant_pebble_1M/reward_model_1000000_0.pt

ENV_ID=Ant-v2
STEPS=1000000
SEEDS=(6 8 9 10)

SID=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$SID]}

echo "Seed=${SEED}"
echo "Dataset: ${FORCE_DATASET}"
echo "Pref RM: ${PREF_RM}"

# ------------------------------
# CPU / headless MuJoCo
# ------------------------------
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

# ------------------------------
# W&B project for Ant
# ------------------------------
RUN_NAME="PAIL_Ant1200_beta5_${STEPS}_s${SEED}"
export WANDB_PROJECT="SAIL_variants_ant"
export WANDB_NAME="${RUN_NAME}"

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
# Run PR-SAIL (Pref-Reweight Teacher)
# IMPORTANT: Ant RM expects obs_dim = 111 (from your eval: obs_dim=111 act_dim=8 rm_in_dim=119)
# ------------------------------
set -x
PYTHONPATH="$PWD/.." python -u train_sail.py \
  --env "${ENV_ID}" \
  --algo sail \
  --task gail-lfd-adaptive-dynamic \
  --seed "${SEED}" \
  --n-timesteps "${STEPS}" \
  --n-episodes "${EPISODES}" \
  --pref-reweight-teacher \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 111 \
  --pref-beta 5.0 \
  --pref-promote-quantile 0.5 \
  --pref-max-teacher-trajs 500 \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "======================================="
echo "Done: ${RUN_NAME}"
date
echo "======================================="