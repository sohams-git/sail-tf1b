#!/bin/bash
#SBATCH --job-name=sail_weighted_tac_unf_ant
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_weighted_tac_unf_ant_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_weighted_tac_unf_ant_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: WEIGHTED-DISC + SOFT-TAC (UNFILTERED PAIRS) | Ant-v2 =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# ---------------- Paths ----------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# ---------------- Experiment knobs ----------------
ENV_ID=Ant-v2
STEPS=1000000
EPISODES=10
SEEDS=(0 2 5)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# ---------------- Ant teacher dataset ----------------
DATASET="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_${EPISODES}.npz"

# ---------------- Offline preference RM (Ant) ----------------
PREF_RM=/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/ant_pebble_1M/reward_model_1000000_0.pt

# ---------------- MuJoCo / perf ----------------
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

# ---------------- Activate env ----------------
source "${VENV}/bin/activate"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"

# ---------------- Safety checks ----------------
if [[ ! -f "${PREF_RM}" ]]; then
  echo "[ERROR] Missing pref reward model: ${PREF_RM}" >&2
  exit 3
fi
if [[ ! -f "${DATASET}" ]]; then
  echo "[ERROR] Missing Ant dataset: ${DATASET}" >&2
  exit 4
fi

echo "Dataset: ${DATASET}"
echo "Resolved dataset: $(readlink -f "${DATASET}")"
ls -lh "${DATASET}"

cd "${RUN_DIR}"

# ---------------- Run name ----------------
RUN_NAME="PAILTAC_Ant1200_${STEPS}_s${SEED}"

# ---------------- W&B (Ant project) ----------------
export WANDB_PROJECT="SAIL_variants_ant"
export WANDB_NAME="${RUN_NAME}"
export WANDB_GROUP="${ENV_ID}_weighted_softTAC_${STEPS}"
export WANDB_SILENT=true

echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "Using python: $(which python)"
python --version

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
  --pref-beta 5.0 \
  --pref-promote-quantile 0.5 \
  --pref-max-teacher-trajs 500 \
  \
  --pref-rank-disc \
  --pref-rank-weight 0.0 \
  --pref-rank-batch-size 32 \
  \
  --pref-soft-rank-weight 0.5 \
  --pref-soft-rank-temp 1.0 \
  \
  --pref-pair-filter none \
  \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 111 \
  --pref-norm zscore \
  --pref-clip 5.0 \
  \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "======================================="
echo "Done: ${RUN_NAME}"
date
echo "======================================="