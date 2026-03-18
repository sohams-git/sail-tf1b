#!/bin/bash
#SBATCH --job-name=sail_weighted_tac_unf_walker
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_weighted_tac_unf_walker_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_weighted_tac_unf_walker_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: WEIGHTED-DISC + SOFT-TAC (UNFILTERED PAIRS) | Walker2d-v2 =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# ---------------- Paths ----------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# ---------------- Experiment knobs ----------------
ENV_ID=Walker2d-v2
STEPS=1000000
EPISODES=10
SEEDS=(3 5 7 8 9 10)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# ---------------- Walker2d teacher dataset ----------------
DATASET="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Walker2d_scores_1500_episodes_${EPISODES}.npz"

# ---------------- Offline preference RM (Walker2d) ----------------
PREF_RM=/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/walker2d_pebble_1M/reward_model_1000000_0.pt

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
  echo "[ERROR] Missing Walker2d dataset: ${DATASET}" >&2
  exit 4
fi

echo "Dataset: ${DATASET}"
echo "Resolved dataset: $(readlink -f "${DATASET}")"
ls -lh "${DATASET}"

cd "${RUN_DIR}"

# ---------------- Run name ----------------
RUN_NAME="PAILTAC_b1_Walker1500_${STEPS}_s${SEED}"

# ---------------- W&B (Walker2d project) ----------------
export WANDB_PROJECT="SAIL_variants_walker2d"
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
  --pref-beta 1.0 \
  --pref-promote-quantile 0.5 \
  --pref-max-teacher-trajs 500 \
  \
  --pref-rank-disc \
  --pref-rank-weight 0.0 \
  --pref-rank-batch-size 32 \
  \
  --pref-soft-rank-weight 0.1 \
  --pref-soft-rank-temp 1.0 \
  \
  --pref-pair-filter none \
  \
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 17 \
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