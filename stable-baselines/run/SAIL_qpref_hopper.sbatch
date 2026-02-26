#!/bin/bash
#SBATCH --job-name=sail_qpref_hopper
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_qpref_hopper_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_qpref_hopper_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: TD3 + Q-preference loss (Qpref) on critic | Hopper-v2 =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# ---------------- Paths ----------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# ---------------- Experiment knobs ----------------
ENV_ID=Hopper-v2
STEPS=1000000
EPISODES=10
SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# ---------------- Qpref hyperparams ----------------
QPREF_W=0.03          # lambda (qpref_weight)
QPREF_TEMP=1.0       # temperature T
QPREF_BSZ=32         # number of pref pairs per critic update
QPREF_START=0        # start step

# ---------------- Offline preference RM (Hopper) ----------------
# Used to compute preference returns J_phi for Qpref pairing.
PREF_RM=/nfs/turbo/umd-sabymath/Soham/BPref/exported_models/hopper_pebble_1M/reward_model_1000000_0.pt

# ---------------- Hopper teacher dataset ----------------
# For Hopper you already have the exact filename SAIL uses, so no symlink needed.
DATASET="${REPO_ROOT}/teacher_dataset/expert_data_no_img_Hopper_scores_1500_episodes_${EPISODES}.npz"

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
  echo "[ERROR] Missing Hopper dataset: ${DATASET}" >&2
  exit 4
fi

echo "Dataset: ${DATASET}"
echo "Resolved dataset: $(readlink -f "${DATASET}")"
ls -lh "${DATASET}"

cd "${RUN_DIR}"

# ---------------- Run name ----------------
RUN_NAME="SAIL_QPREF_Hopper1500_${STEPS}_s${SEED}"

# ---------------- W&B (Hopper project) ----------------
export WANDB_PROJECT="SAIL_variants_hopper"
export WANDB_NAME="${RUN_NAME}"
export WANDB_GROUP="${ENV_ID}_qpref_${STEPS}"
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
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 11 \
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