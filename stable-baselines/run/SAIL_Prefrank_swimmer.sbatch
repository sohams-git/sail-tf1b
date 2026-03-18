#!/bin/bash
#SBATCH --job-name=sail_pref_rank_only_swimmer
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_pref_rank_only_swimmer_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_pref_rank_only_swimmer_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: Preference Ranking ONLY (Swimmer) (no teacher reweight) =="
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
# Experiment setup
# ------------------------------
ENV_ID=Swimmer-v2
STEPS=1000000
EPISODES=4

SEEDS=(1 3 4 5 6)
SID=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$SID]}

# ------------------------------
# Dataset (force exact file SAIL expects)
# ------------------------------
FORCE_DATASET="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Swimmer_scores_120_episodes_${EPISODES}.npz"

# ------------------------------
# Swimmer Preference RM
# ------------------------------
PREF_RM="/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/swimmer_pebble_1M/reward_model_1000000_0.pt"

# ------------------------------
# MuJoCo / perf
# ------------------------------
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

# ------------------------------
# Activate env
# ------------------------------
source "${VENV}/bin/activate"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"
cd "${RUN_DIR}"

echo "Using python: $(which python)"
python --version

# ------------------------------
# W&B (Swimmer project)
# ------------------------------
RUN_NAME="SAIL_PrefRankOnly0.05_Swimmer120_${STEPS}_s${SEED}"

export WANDB_PROJECT="SAIL_variants_swimmer"
export WANDB_NAME="${RUN_NAME}"
export WANDB_GROUP="${ENV_ID}_pref_rank_only_${STEPS}"

echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"

# ------------------------------
# Safety checks
# ------------------------------
if [[ ! -f "${PREF_RM}" ]]; then
  echo "[ERROR] Missing pref reward model: ${PREF_RM}" >&2
  exit 3
fi

if [[ ! -f "${FORCE_DATASET}" ]]; then
  echo "[ERROR] Missing dataset: ${FORCE_DATASET}" >&2
  exit 4
fi

# ------------------------------
# Dataset symlink
# ------------------------------
TARGET="../teacher_dataset/expert_data_no_img_Swimmer_scores_120_episodes_${EPISODES}.npz"

TARGET_ABS="$(readlink -f "${TARGET}" 2>/dev/null || echo "")"
FORCE_ABS="$(readlink -f "${FORCE_DATASET}")"

if [[ "${TARGET_ABS}" == "${FORCE_ABS}" ]]; then
  echo "[INFO] Dataset already correct."
else
  rm -f "${TARGET}"
  ln -s "${FORCE_DATASET}" "${TARGET}"
fi

echo "Resolved dataset: $(readlink -f "${TARGET}" || echo MISSING)"

# ------------------------------
# Run
# Swimmer RM expects obs_dim = 8
# ------------------------------
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
  --pref-rank-weight 0.05 \
  --pref-rank-batch-size 32 \
  \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 8 \
  \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "Done: ${RUN_NAME}"
date