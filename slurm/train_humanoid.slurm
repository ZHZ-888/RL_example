#!/bin/bash -e

#SBATCH --job-name=humanoid_ppo
#SBATCH --account=uoa04480
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=32G
#SBATCH --output=/nesi/project/uoa04480/rl_agent_training_example/slurm_logs/%x_%j.out
#SBATCH --error=/nesi/project/uoa04480/rl_agent_training_example/slurm_logs/%x_%j.err

module purge
module load Python/3.11.3-gimkl-2022a
source /nesi/project/uoa04480/myenv/bin/activate
export PYTHONNOUSERSITE=1

# Prevent each subprocess from starting additional BLAS/OpenMP thread pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Override these at submission time with: sbatch --export=ALL,SEED=1,TOTAL_TIMESTEPS=100000 ...
SEED="${SEED:-42}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
NUM_ENVS="${SLURM_CPUS_PER_TASK}"

OUTPUT_ROOT="/nesi/project/uoa04480/rl_agent_training_example/outputs"
RUN_DIR="${OUTPUT_ROOT}/${SLURM_JOB_ID}_seed${SEED}"
CODE_DIR="${SLURM_SUBMIT_DIR}"

mkdir -p "${RUN_DIR}"
cd "${CODE_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Code directory: ${CODE_DIR}"
echo "Run directory: ${RUN_DIR}"
echo "Python: $(which python)"
echo "Seed: ${SEED}"
echo "Parallel environments: ${NUM_ENVS}"
echo "Total timesteps: ${TOTAL_TIMESTEPS}"

srun python -u train_humanoid_sb3.py \
    --output-dir "${RUN_DIR}" \
    --num-envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --total-timesteps "${TOTAL_TIMESTEPS}"
