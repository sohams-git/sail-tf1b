#!/bin/bash
#SBATCH --job-name=sail_pref_rm_rank_hc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_pref_rm_rank_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_pref_rm_rank_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL: Preference RM-Ranking loss (new pref_rm_loss) =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

ENV_ID=HalfCheetah-v2
STEPS=500000
EPISODES=4
SEEDS=(1)
SID=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$SID]}

# Reward model (BPref export)
PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

# CPU MuJoCo
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_CPP_MIN_LOG_LEVEL=2

source "${VENV}/bin/activate"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"
cd "${RUN_DIR}"

RUN_NAME="SAIL_pref_rm_rank_${STEPS}_s${SEED}"

export WANDB_PROJECT="SAIL_variants"
export WANDB_NAME="${RUN_NAME}"

set -x
PYTHONPATH="$PWD/.." python -u train_sail.py \
  --env "${ENV_ID}" \
  --algo sail \
  --task gail-lfd-adaptive-dynamic \
  --seed "${SEED}" \
  --n-timesteps "${STEPS}" \
  --n-episodes "${EPISODES}" \
  \
  # ---- NEW RM preference loss (your pref_rm_loss) ----
  --pref-rm "${PREF_RM}" \
  --pref-expect-obs-dim 17 \
  --pref-rm-rank \
  --pref-rm-rank-weight 0.1 \
  --pref-rm-rank-batch-size 16 \
  \
  # ---- Turn OFF disc-based preference loss (optional) ----
  --pref-rank-weight 0.0 \
  \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir        "logs/${RUN_NAME}"
set +x

echo "Done: ${RUN_NAME}"
date