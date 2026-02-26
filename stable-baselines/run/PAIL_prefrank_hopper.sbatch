#!/bin/bash
#SBATCH --job-name=sail_prefrew_rank_hopper
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_prefrew_rank_hopper_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_prefrew_rank_hopper_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: Pref-Reweight + Pref-Rank (Hopper) =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# ---------------- Paths ----------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

ENV_ID=Hopper-v2
STEPS=1000000
EPISODES=10
SEEDS=(0 1 2 3 4)
SID=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$SID]}

# ---------------- Hopper Reward Model ----------------
PREF_RM=/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/hopper_pebble_1M/reward_model_1000000_0.pt

# ---------------- Hopper Dataset ----------------
FORCE_DATASET=${REPO_ROOT}/teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_${EPISODES}.npz

# ---------------- MuJoCo / GPU render ----------------
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

# ---------------- Activate env ----------------
source "${VENV}/bin/activate"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"
cd "${RUN_DIR}"

echo "Using python: $(which python)"
python --version

# ---------------- Safety checks ----------------
if [[ ! -f "${PREF_RM}" ]]; then
  echo "ERROR: Missing pref reward model: ${PREF_RM}" >&2
  exit 3
fi

if [[ ! -f "${FORCE_DATASET}" ]]; then
  echo "ERROR: Missing dataset: ${FORCE_DATASET}" >&2
  exit 4
fi

# ---------------- Dataset symlink (safe) ----------------
TARGET="../teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_${EPISODES}.npz"

TARGET_ABS="$(readlink -f "${TARGET}" 2>/dev/null || echo "")"
FORCE_ABS="$(readlink -f "${FORCE_DATASET}")"

if [[ "${TARGET_ABS}" != "${FORCE_ABS}" ]]; then
  rm -f "${TARGET}"
  ln -s "${FORCE_DATASET}" "${TARGET}"
fi

echo "Resolved dataset: $(readlink -f ${TARGET})"

# ---------------- Run naming ----------------
RUN_NAME="PAIL_PrefRank_Hopper1500_${STEPS}_s${SEED}"

export WANDB_PROJECT="SAIL_variants_hopper"
export WANDB_NAME="${RUN_NAME}"
export WANDB_GROUP="${ENV_ID}_prefrew_rank_${STEPS}"

echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"

# ---------------- Training ----------------
set -x
PYTHONPATH="$PWD/.." python -u train_sail.py \
  --env "${ENV_ID}" \
  --algo sail \
  --task gail-lfd-adaptive-dynamic \
  --seed "${SEED}" \
  --n-timesteps "${STEPS}" \
  --n-episodes "${EPISODES}" \
  \
  --pref-reweight-teacher \
  --pref-rank-disc \
  --pref-rank-weight 0.03 \
  --pref-rank-batch-size 32 \
  \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 11 \
  --pref-beta 1.0 \
  --pref-promote-quantile 0.5 \
  --pref-max-teacher-trajs 500 \
  \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "Done: ${RUN_NAME}"
date