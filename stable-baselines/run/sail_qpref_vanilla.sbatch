#!/bin/bash
#SBATCH --job-name=sail_qpref_hc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_qpref_%j.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_qpref_%j.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: TD3 + Q-preference loss (Qpref) on critic =="
date
hostname
echo "JOB=$SLURM_JOB_ID"

# ---------------- Paths ----------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

ENV_ID=HalfCheetah-v2
STEPS=500000
EPISODES=4
SEED=1

# Qpref hyperparams
QPREF_W=0.1          # lambda (qpref_weight)
QPREF_TEMP=1.0       # temperature T
QPREF_BSZ=32         # number of pref pairs per critic update
QPREF_START=0        # start step

# Offline preference RM (BPref export) — used to compute J_phi for teacher episodes
PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

# Teacher dataset
ORIG_DATASET="${REPO_ROOT}/stable-baselines/teacher_dataset/legacy/expert_data_no_img_HalfCheetah_scores_500_episodes_${EPISODES}_legacy_exact.npz"
TARGET_LINK="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_${EPISODES}.npz"

# ------------- Env (CPU MuJoCo) -------------
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

# ------------- Activate env -------------
source "${VENV}/bin/activate"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"

# ------------- Dataset link -------------
cd "${REPO_ROOT}"
mkdir -p teacher_dataset
ln -sfn "${ORIG_DATASET}" "${TARGET_LINK}"

cd "${RUN_DIR}"

# ---------------- Run name ----------------
RUN_NAME="SAIL_QPREF_w${QPREF_W}_T${QPREF_TEMP}_B${QPREF_BSZ}_${STEPS}_s${SEED}"

export WANDB_PROJECT="SAIL_variants"
export WANDB_NAME="${RUN_NAME}"
export WANDB_SILENT=true

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
  --pref-expect-obs-dim 17 \
  \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "Done: ${RUN_NAME}"
date