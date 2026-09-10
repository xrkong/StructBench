"""Benchmark card for vehicle_barrier_crash_3d -- private (ADR-0058)."""

from ..card import BenchmarkCard
from .benchmark import AUX_FIELD, QOIS, TEST, TRAIN, VAL

CARD = BenchmarkCard(
    name="VehicleBarrierCrash-3D",
    version="0.1",
    description=(
        "PRIVATE (ADR-0058): LS-DYNA vehicle-vs-barrier crash surrogate. Real "
        "data (New_Road_Barrier family, varying layer configuration), not "
        "distributable -- never published, no blessed/provisional results."
    ),
    provenance=(
        "Private LS-DYNA vehicle-barrier crash sweep (New_Road_Barrier family); "
        "downsampled to canonical nodes by collider's own region-aware FPS "
        "pipeline before conversion into this schema."
    ),
    data_license="private, not distributable (ADR-0058)",
    solver="LS-DYNA",
    discretisation="FEM",
    materials=(
        "MAT_CONCRETE_DAMAGE_REL3",
        "MAT_PIECEWISE_LINEAR_PLASTICITY",
        "MAT_MODIFIED_PIECEWISE_LINEAR_PLASTICITY",
        "MAT_RIGID",
        "MAT_BLATZ-KO_RUBBER",
        "MAT_LOW_DENSITY_FOAM",
        "MAT_SPOTWELD",
        "MAT_SPRING_ELASTIC/NONLINEAR_ELASTIC",
    ),
    erosion=True,
    loading="vehicle impact into a W-beam/concrete road barrier at a fixed speed",
    source_units="t-mm-s",
    geometry=(
        "vehicle body + New_Road_Barrier (W-beam/concrete), region-aware FPS "
        "downsampled to 100,046 nodes/case (collider's ~100k budget)"
    ),
    n_cases=len(TRAIN) + len(VAL) + len(TEST),
    splits={
        "train": len(TRAIN),
        "val": len(VAL),
        "test": len(TEST),
    },
    task="autoregressive or time-conditioned transition (undecided, see model-spec)",
    aux_field=AUX_FIELD,
    aux_unit="dimensionless",
    qois=tuple(QOIS),
    fields=(
        "node/displacement",
        "node/effective_plastic_strain",
    ),
    particles_per_case="100046-100046",
    n_frames=50,
    output_dt_ms=20.0,
    input_frames=2,
    protocol_rationale=(
        "PLACEHOLDER: no ground-truth timeline analysis has been run (ADR-0032 "
        "§5 requires one before a real protocol is pinned). input_frames=2 is "
        "a conservative stand-in pending that analysis -- 50 total frames over "
        "~980ms means there is little headroom to spend on a longer seed."
    ),
    size_gb=None,
)
