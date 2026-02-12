#!/bin/bash
#SBATCH -A dssc
#SBATCH -p EPYC
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-6
#SBATCH --job-name=bovw_array
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

# ---------------------------------
# Folders
# ---------------------------------
mkdir -p logs
# mkdir -p results/runs

# ---------------------------------
# Variant selection (array)
# ---------------------------------
case "${SLURM_ARRAY_TASK_ID}" in
  1) VAR="1_BoVW_Hard_kNN" ;;
  2) VAR="2_BoVW_Hard_LinearSVC" ;;
  3) VAR="3_BoVW_Hard_Chi2SVC" ;;
  4) VAR="4_BoVW_Hard_ECOC" ;;
  5) VAR="5_BoVW_Soft_SVC" ;;
  6) VAR="6_PMK" ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

# ---------------------------------
# Experiment parameters
# ---------------------------------
DATA_ROOT="data"
CV_SPLITS=5
SCORER="f1_macro"
# KS="32,64,128,256,512"
KS="512"
OUT_ROOT=$(pwd)
SEED=0
N_JOBS=-1

# ---------------------------------
# Activate environment
# ---------------------------------
source ~/CVPR/cvenv/bin/activate

# ---------------------------------
# Threading / BLAS controls
# ---------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ---------------------------------
# Run directory & logging
# ---------------------------------
RUN_ID="$(date +%Y%m%d_%H%M)"
RUN_DIR="${OUT_ROOT}/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
VAR_DIR="${RUN_DIR}/${VAR}"
mkdir -p "${VAR_DIR}"

echo "========================================"
echo " BoVW SLURM job"
echo "----------------------------------------"
echo "Job ID        : ${SLURM_JOB_ID:-NA}"
echo "Array Job ID  : ${SLURM_ARRAY_JOB_ID:-NA}"
echo "Array Task ID : ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node          : ${SLURM_NODELIST:-NA}"
echo "Partition     : ${SLURM_JOB_PARTITION:-NA}"
echo "CPUs/task     : ${SLURM_CPUS_PER_TASK:-NA}"
echo "Memory/node   : ${SLURM_MEM_PER_NODE:-NA}"
echo "Variant       : ${VAR}"
echo "Run directory : ${RUN_DIR}"
echo "Run variant directory : ${VAR_DIR}"
echo "========================================"
echo

python -V
which python
echo

# ---------------------------------
# Save run metadata
# ---------------------------------
cat > "${VAR_DIR}/meta.txt" << EOF
time: $(date -Is)
job_id: ${SLURM_JOB_ID:-NA}
array_job_id: ${SLURM_ARRAY_JOB_ID:-NA}
array_task_id: ${SLURM_ARRAY_TASK_ID:-NA}
node: ${SLURM_NODELIST:-NA}
partition: ${SLURM_JOB_PARTITION:-NA}
cpus_per_task: ${SLURM_CPUS_PER_TASK:-NA}
mem_per_node: ${SLURM_MEM_PER_NODE:-NA}
python: $(which python)
variant: ${VAR}
data_root: ${DATA_ROOT}
cv_splits: ${CV_SPLITS}
scorer: ${SCORER}
ks: ${KS}
seed: ${SEED}
n_jobs: ${N_JOBS}
EOF

# ---------------------------------
# Mirror stdout/stderr into run.log
# ---------------------------------
exec > >(tee -a "${VAR_DIR}/run.log") 2>&1

# ---------------------------------
# Cleanup handler (for clarity in logs)
# ---------------------------------
cleanup() {
  echo
  echo "----------------------------------------"
  echo "Job finished at $(date -Is)"
  echo "----------------------------------------"
}
trap cleanup EXIT INT TERM

# ---------------------------------
# Run experiment
# ---------------------------------
echo
echo ">>> Running BoVW experiment: ${VAR}"
echo

python run_bovw_experiments.py \
  --data_root "${DATA_ROOT}" \
  --variant "${VAR}" \
  --run_id "${RUN_DIR}" \
  --cv_splits "${CV_SPLITS}" \
  --scorer "${SCORER}" \
  --ks "${KS}" \
  --out_root "${OUT_ROOT}" \
  --seed "${SEED}" \
  --n_jobs "${N_JOBS}"

echo
echo "✅ DONE"
echo "Results root : ${OUT_ROOT}"
echo "Run folder   : ${VAR_DIR}"

