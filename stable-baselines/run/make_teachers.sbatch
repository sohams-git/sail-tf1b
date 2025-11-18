#!/bin/bash
#SBATCH --job-name=hc_teachers_cpu
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/turbo/umd-sridas/Soham/SAIL-tf1b/stable-baselines/run/logs/teach_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sridas/Soham/SAIL-tf1b/stable-baselines/run/logs/teach_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== HC teachers (CPU) =="
date; hostname
echo "JOB=$SLURM_JOB_ID  TASK=$SLURM_ARRAY_TASK_ID"

# -------- Paths (sridas) --------
REPO=/nfs/turbo/umd-sridas/Soham/SAIL-tf1b
RUN_DIR=${REPO}/stable-baselines/run
VENV=$HOME/.venvs/sb-tf1-clean

# All outputs go here:
SAVE_ROOT="${REPO}/stable-baselines/teacher_dataset/mix_creation"
mkdir -p "${SAVE_ROOT}" "${RUN_DIR}/logs"

# Activate env
source "$VENV/bin/activate"

# Headless MuJoCo (CPU)
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

cd "$REPO"

# Which teacher this task runs
EPISODES=(50 150 300 500)
TAG=${EPISODES[$SLURM_ARRAY_TASK_ID]}
N_TRAJS=25
SEED=0

# Unique run folder under mix_creation
RUN_TAG="run_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="${SAVE_ROOT}/${RUN_TAG}"
mkdir -p "$SAVE_DIR"
export SAVE_DIR

echo "Training: episodes=${TAG}  n_trajs=${N_TRAJS}  seed=${SEED}"
echo "Saving to: ${SAVE_DIR}"

# (Optional) If you patched the trainer to be verbose/loggy, you'll see progress here
PYTHONHASHSEED=0 python -u stable-baselines/scripts/make_halfcheetah_teachers_tf1.py \
  --episodes "${TAG}" \
  --n_trajs "${N_TRAJS}" \
  --save_dir "${SAVE_DIR}" \
  --seed "${SEED}"

echo "== Written files in ${SAVE_DIR} =="
ls -lh "${SAVE_DIR}" || true

# Quick stats of the NPZ files
python - <<'PY'
import os, glob, numpy as np
save_dir = os.environ["SAVE_DIR"]
paths = sorted(glob.glob(os.path.join(save_dir, "*.npz")))
if not paths:
    print("No NPZ yet; trainer may still be running.")
for p in paths:
    d = np.load(p, allow_pickle=True)
    R = d["episode_returns"].astype(float)
    print(os.path.basename(p),
          "episodes=", len(R),
          "mean/min/max=", f"{R.mean():.1f}/{R.min():.1f}/{R.max():.1f}")
PY

# Convenience: update "latest" symlink to this run
ln -sfn "$SAVE_DIR" "${SAVE_ROOT}/latest"

echo "DONE."
date