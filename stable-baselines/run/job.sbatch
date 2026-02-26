#!/bin/bash
#SBATCH --job-name=sail_cpu_all3
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0                          # 3 variants = 3 tasks
#SBATCH --output=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_%A_%a.out
#SBATCH --error=/nfs/turbo/umd-sabymath/Soham/sail-tf1b/stable-baselines/run/logs/slurm_%A_%a.err
#SBATCH --account=sabymath0

set -euo pipefail
echo "== SAIL CPU run =="
date
hostname
echo "JOB=$SLURM_JOB_ID TASK=$SLURM_ARRAY_TASK_ID"

# -------- EDIT ME (paths you control) ---------------------------------
REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b
RUN_DIR=${REPO_ROOT}/stable-baselines/run
VENV=$HOME/.venvs/sail-tf1b

# Teacher dir that should appear as ../teacher_dataset from RUN_DIR
TEACHER_SRC=${REPO_ROOT}/teacher_dataset

# Choose the teacher dataset file you want this run to use:
# (example: your 3000-episode legacy-exact file)
FORCE_DATASET="${REPO_ROOT}/stable-baselines/teacher_dataset/legacy/expert_data_no_img_HalfCheetah_scores_500_episodes_4_legacy_exact.npz"
EPISODES=4   # must match the suffix in the filename SAIL builds
# ----------------------------------------------------------------------

# (Only used for non-vanilla variants)
PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

ENV_ID=HalfCheetah-v2
STEPS=1000000      # quick run; increase for full training
SEEDS=(5)
# VARIANTS=(vanilla critic adddisc)
# VARIANTS=(adddisc)
VARIANTS=(adddisc)

VID=$(( SLURM_ARRAY_TASK_ID % ${#VARIANTS[@]} ))
SID=$(( SLURM_ARRAY_TASK_ID / ${#VARIANTS[@]} ))
VAR=${VARIANTS[$VID]}
SEED=${SEEDS[$SID]:-0}

echo "Variant=${VAR}  Seed=${SEED}"

# ---- CPU / headless MuJoCo ---------------------------------------------------
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

# ---- Env + paths -------------------------------------------------------------
source "${VENV}/bin/activate"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/tb_logs"

# Make sure we are using the dedicated SAIL hyperparams (not TD3)
cd "${RUN_DIR}"
if [[ ! -f ../hyperparams/sail.yml ]]; then
  echo "[ERROR] ../hyperparams/sail.yml is missing. Please restore it (git pull) before running."
  exit 1
fi
echo "[INFO] Using hyperparams file: $(readlink -f ../hyperparams/sail.yml)"

# Ensure ../teacher_dataset exists and points to your chosen teacher dir
cd "${REPO_ROOT}"
ln -sfn "${TEACHER_SRC}" teacher_dataset

# Force-map SAIL’s expected filename to the chosen dataset
# SAIL will open: ../teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_${EPISODES}.npz
TARGET_LINK="${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_${EPISODES}.npz"
if [[ -n "${FORCE_DATASET}" ]]; then
  ln -sfn "${FORCE_DATASET}" "${TARGET_LINK}"
fi
echo "Dataset link:"
ls -l "${TARGET_LINK}" || true
echo "Resolved to: $(readlink -f "${TARGET_LINK}" || echo MISSING)"

cd "${RUN_DIR}"

COMMON_ARGS=(
  --env "${ENV_ID}"
  --algo sail
  --task gail-lfd-adaptive-dynamic
  --seed "${SEED}"
  --n-timesteps "${STEPS}"
  --n-episodes "${EPISODES}"
)

case "${VAR}" in
  vanilla)
    RUN_NAME="vanillaSAIL300ep_${STEPS}_s${SEED}"
    EXTRA_ARGS=()
    ;;

  critic)
    RUN_NAME="bpref_criticC_50_hc_v2_${STEPS}_s${SEED}"
    EXTRA_ARGS=( --use-pref-critic --pref-rm "${PREF_RM}" --pref-expect-obs-dim 17 --lambda-ref 0.10 )
    ;;

  adddisc)
    RUN_NAME="discsail500_${STEPS}_s${SEED}"
    EXTRA_ARGS=( --add-pref-to-disc --pref-rm "${PREF_RM}" --pref-expect-obs-dim 17 --pref-weight 1.0 --pref-norm zscore --pref-clip 5.0 )
    ;;

  *)
    echo "Unknown variant ${VAR}" >&2; exit 2;;
esac

# ---- W&B naming from sbatch (Option B) ----
export WANDB_PROJECT="SAIL_variants"
export WANDB_NAME="${RUN_NAME}"
# export WANDB_GROUP="${ENV_ID}_${VAR}_${STEPS}"
# Optional: your account/entity (only if needed)
# export WANDB_ENTITY="sohams1-university-of-michigan"
# ------------------------------------------

# Safety check for pref model when required
if [[ "${VAR}" != "vanilla" ]] && [[ ! -f "${PREF_RM}" ]]; then
  echo "Missing pref reward model: ${PREF_RM}" >&2
  exit 3
fi

set -x
PYTHONPATH="$PWD/.." python -u train_sail.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --tensorboard-log "tb_logs/${RUN_NAME}" \
  --log-dir      "logs/${RUN_NAME}"
set +x

echo "Done: ${RUN_NAME}"
date