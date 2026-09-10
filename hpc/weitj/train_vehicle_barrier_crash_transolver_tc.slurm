#!/bin/bash
# vehicle_barrier_crash_3d Transolver TIME-CONDITIONED training on weitj
# (dgx-b200-1, SLURM partition LocalQ) — ADR-0054/0058.
#
# PRIVATE benchmark: results never get blessed/published (see
# benchmarks/vehicle_barrier_crash_3d/__init__.py). configs/vehicle_barrier_
# crash_3d/transolver.toml still carries several placeholder values
# (input_frames/train_frames pending a real ground-truth timeline analysis;
# training_steps/batch_size copied from DeformingPlate's recipe, 45x fewer
# nodes) -- run hpc/weitj/probe_vehicle_barrier_crash_transolver.slurm FIRST
# and size STEPS below from its real steps/s, rather than trusting the
# config file's placeholder training_steps blindly.
#
# Submit from the repo root, from a login/interactive shell on this node:
#   mkdir -p scratch/logs
#   STEPS=5000 sbatch hpc/weitj/train_vehicle_barrier_crash_transolver_tc.slurm
#
# Only 1/8 GPUs was free at write time (squeue showed 2 running xangruik
# dot-train jobs + 2 zifans jupyter jobs holding the rest) -- single-GPU
# request, matches every other Transolver recipe in this repo.

#SBATCH --account=proj_iim1
#SBATCH --job-name=vbc3d-transolver-tc
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --output=scratch/logs/slurm-vbc3d-transolver-tc-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source /raid/proj_iim1/xrkong/venvs/structbench-gpu/bin/activate
export PYTHONPATH=src

DATA=${DATA:-/raid/proj_iim1/xrkong/canonical/vehicle_barrier_crash_3d}
OUT=${OUT:-/raid/proj_iim1/xrkong/runs/vbc3d-transolver-tc-$SLURM_JOB_ID}
STEPS=${STEPS:-}  # required: set from the probe job's measured steps/s

if [ -z "$STEPS" ]; then
    echo "error: set STEPS (e.g. STEPS=5000 sbatch ...) -- sized from" \
         "hpc/weitj/probe_vehicle_barrier_crash_transolver.slurm's measured" \
         "steps/s, not guessed"
    exit 2
fi
mkdir -p "$OUT"

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

CFG=$(mktemp --suffix=.toml)
sed "s/^training_steps = .*/training_steps = ${STEPS}/" \
    configs/vehicle_barrier_crash_3d/transolver.toml > "$CFG"

# train -> model-best-<step>.pt (val rollout position RMSE over the 1 val
# case's scored span) + periodic ckpt-<step>.pt + config.json.
python -m structbench.cli.train --mode train \
    --config "$CFG" --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/train.log"
# eval on val + test: metrics-*.json + rollouts/*.npz (architecture rebuilt
# from the run's own config.json). If train is killed these never ran
# (set -e) -- rerun just these two lines against the same --out.
python -m structbench.cli.train --mode valid \
    --data-root "$DATA" --out "$OUT" 2>&1 | tee "$OUT/eval.log"
python -m structbench.cli.train --mode rollout \
    --data-root "$DATA" --out "$OUT" 2>&1 | tee -a "$OUT/eval.log"
