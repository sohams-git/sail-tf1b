#!/bin/bash
#SBATCH --job-name=sail_weighted_tac_unf_swimmer
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_weighted_tac_unf_swimmer_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_weighted_tac_unf_swimmer_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: WEIGHTED-DISC + SOFT-TAC (UNFILTERED PAIRS) | Swimmer-v2 =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# ---------------- Paths ----------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# ---------------- Experiment knobs ----------------
ENV_ID=Swimmer-v2
STEPS=1000000
EPISODES=4
SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# ---------------- Swimmer teacher dataset ----------------
DATASET="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Swimmer_scores_120_episodes_${EPISODES}.npz"

# ---------------- Offline preference RM (Swimmer) ----------------
PREF_RM=/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/swimmer_pebble_1M/reward_model_1000000_0.pt

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
  echo "[ERROR] Missing Swimmer dataset: ${DATASET}" >&2
  exit 4
fi

echo "Dataset: ${DATASET}"
echo "Resolved dataset: $(readlink -f "${DATASET}")"
ls -lh "${DATASET}"

cd "${RUN_DIR}"

# ---------------- Run name ----------------
RUN_NAME="PAILTAC_Swimmer120_${STEPS}_s${SEED}"

# ---------------- W&B (Swimmer project) ----------------
# NOTE: if your code hardcodes the project to SAIL_variants, this env var may not override it.
export WANDB_PROJECT="SAIL_variants_swimmer"
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
  --pref-beta 50.0 \
  --pref-promote-quantile 0.5 \
  --pref-max-teacher-trajs 500 \
  \
  --pref-rank-disc \
  --pref-rank-weight 0.0 \
  --pref-rank-batch-size 32 \
  \
  --pref-soft-rank-disc \
  --pref-soft-rank-weight 0.5 \
  --pref-soft-rank-temp 1.0 \
  \
  --pref-pair-filter none \
  \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 8 \
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