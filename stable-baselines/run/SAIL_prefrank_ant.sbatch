#!/bin/bash
#SBATCH --job-name=sail_pref_rank_only_ant
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_pref_rank_only_ant_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_pref_rank_only_ant_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: Preference Ranking ONLY (Ant) (no teacher reweight) =="
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
ENV_ID=Ant-v2
STEPS=1000000
EPISODES=10
SEEDS=(0 1 3 4 5)
SID=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$SID]}

# ------------------------------
# Dataset (force exact file SAIL expects)
# SAIL opens: ../teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_${EPISODES}.npz
# ------------------------------
FORCE_DATASET="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_10.npz"

# ------------------------------
# Ant Preference RM (.pt state_dict)
# ------------------------------
PREF_RM="/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/ant_pebble_1M/reward_model_1000000_0.pt"

# ------------------------------
# MuJoCo / perf (GPU-render headless)
# ------------------------------
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

# ------------------------------
# Activate env + dirs
# ------------------------------
source "${VENV}/bin/activate"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"
cd "${RUN_DIR}"

echo "Using python: $(which python)"
python --version

# ------------------------------
# W&B (Ant project)
# ------------------------------
RUN_NAME="SAIL_PrefRankOnly_Ant1200_${STEPS}_s${SEED}"
export WANDB_PROJECT="SAIL_variants_ant"
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
# Force dataset symlink (avoid ln-to-self issue)
# Only link if FORCE_DATASET != target path
# ------------------------------
TARGET="../teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_${EPISODES}.npz"
echo "Target dataset path: ${TARGET}"
echo "Forcing dataset: ${FORCE_DATASET}"

TARGET_ABS="$(readlink -f "${TARGET}" 2>/dev/null || echo "")"
FORCE_ABS="$(readlink -f "${FORCE_DATASET}")"

if [[ "${TARGET_ABS}" == "${FORCE_ABS}" ]]; then
  echo "[INFO] Dataset already points to FORCE_DATASET (no relink needed)."
else
  rm -f "${TARGET}"
  ln -s "${FORCE_DATASET}" "${TARGET}"
fi

echo "Resolved dataset: $(readlink -f "${TARGET}" || echo MISSING)"

# ------------------------------
# Run
# Notes:
# - Ant RM expects obs_dim = 111 (from your eval: obs_dim=111 act_dim=8 rm_in_dim=119)
# - This variant uses pref ranking loss only (no add-to-disc reward, no teacher reweight)
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
  --pref-rank-weight 0.1 \
  --pref-rank-batch-size 32 \
  \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 111 \
  \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "Done: ${RUN_NAME}"
date