#!/bin/bash
#SBATCH --job-name=sail_weighted_qpref_ant
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_weighted_qpref_ant_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_weighted_qpref_ant_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== Weighted SAIL (teacher reweight) + Qpref (TD3 critic ranking) | Ant-v2 =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# ---------------- Paths ----------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# ---------------- Experiment ----------------
ENV_ID=Ant-v2
STEPS=1000000
EPISODES=10
SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# ---------------- Qpref ----------------
QPREF_W=0.1
QPREF_TEMP=1.0
QPREF_BSZ=32
QPREF_START=0

# ---------------- Preference-weighted SAIL ----------------
# (teacher buffer is reweighted by exp(beta * J_phi))
PREF_BETA=5.0
PREF_PROMOTE_Q=0.5          # keep consistent with your PR-SAIL runs (optional, but recommended)
PREF_MAX_TRAJ=500

# ---------------- Offline Reward Model (Ant) ----------------
PREF_RM=/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/ant_pebble_1M/reward_model_1000000_0.pt

# ---------------- Ant teacher dataset ----------------
DATASET="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Ant_scores_1200_episodes_${EPISODES}.npz"

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

echo "Seed=${SEED}"
echo "Dataset: ${DATASET}"
echo "Resolved dataset: $(readlink -f "${DATASET}")"
echo "Pref RM: ${PREF_RM}"

cd "${RUN_DIR}"

# ---------------- Run name ----------------
RUN_NAME="WeightedQPREF_Ant1200_${STEPS}_s${SEED}"

# ---------------- W&B (Ant project) ----------------
export WANDB_PROJECT="SAIL_variants_ant"
export WANDB_NAME="${RUN_NAME}"
export WANDB_GROUP="${ENV_ID}_weighted_qpref_${STEPS}"
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
  --qpref \
  --qpref-weight "${QPREF_W}" \
  --qpref-temp "${QPREF_TEMP}" \
  --qpref-batch-size "${QPREF_BSZ}" \
  --qpref-start-step "${QPREF_START}" \
  \
  --pref-reweight-teacher \
  --pref-beta "${PREF_BETA}" \
  --pref-promote-quantile "${PREF_PROMOTE_Q}" \
  --pref-max-teacher-trajs "${PREF_MAX_TRAJ}" \
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