#!/bin/bash
# vehicle_barrier_crash_3d Transolver TC — THROUGHPUT PROBE, not a real run
# (ADR-0058). Cheap: 20 pure-training steps (no validation rollout), no
# checkpoint kept beyond the probe dir. Purpose: measure real steps/s and
# peak GPU memory at full recipe capacity (hidden_dim=128, n_layers=8,
# batch_size=2, N=100,046 nodes/case) on a real B200, so
# configs/vehicle_barrier_crash_3d/transolver.toml's placeholder
# training_steps/batch_size can be sized from real numbers instead of
# copied from DeformingPlate's recipe (45x smaller node count).
#
# Submit from the repo root, from a LOGIN/interactive shell on this node
# (dgx-b200-1, partition LocalQ):
#   mkdir -p scratch/logs && sbatch hpc/weitj/probe_vehicle_barrier_crash_transolver.slurm
#
# Only 1/8 GPUs was free at write time (squeue showed 2 running xangruik
# dot-train jobs + 2 zifans jupyter jobs holding the rest) -- single-GPU
# request, matches every other Transolver recipe in this repo (the training
# loop is single-GPU by design).

#SBATCH --account=proj_iim1
#SBATCH --job-name=probe
#SBATCH --partition=LocalQ
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=scratch/logs/slurm-vbc3d-probe-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source /raid/proj_iim1/xrkong/venvs/structbench-gpu/bin/activate
export PYTHONPATH=src

export DATA=${DATA:-/raid/proj_iim1/xrkong/canonical/vehicle_barrier_crash_3d}
export OUT=${OUT:-/raid/proj_iim1/xrkong/runs/vbc3d-probe-$SLURM_JOB_ID}

echo "code commit: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python3 - <<'PYEOF'
import os, time
from pathlib import Path
import torch
from structbench.benchmarks import get_benchmark
from structbench.config import TransolverConfig, TrainConfig
from structbench.cli.train import train

spec = get_benchmark("vehicle_barrier_crash_3d")
cfg = TransolverConfig(
    input_frames=2, dim=3, hidden_dim=128, n_layers=8, n_heads=8, slice_num=64,
    node_type_size=2, noise_std=0.0, normalizer_warmup_steps=100,
    history_frames=0, frames_per_call=1, time_conditioned=True,
)
tcfg = TrainConfig(
    benchmark="vehicle_barrier_crash_3d", batch_size=2,
    training_steps=20, val_every=10_000, train_frames=0,
)
data_root = Path(os.environ["DATA"])
out = Path(os.environ["OUT"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")
t0 = time.time()
ckpt = train(spec, cfg, tcfg, data_root, out, device, family="transolver")
dt = time.time() - t0
print(f"PROBE RESULT: {dt:.1f}s for 20 pure-train steps -> {dt/20:.3f} s/step "
      f"(batch_size=2, N=100046, hidden_dim=128/n_layers=8/n_heads=8/slice_num=64)")
if device == "cuda":
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
PYEOF
