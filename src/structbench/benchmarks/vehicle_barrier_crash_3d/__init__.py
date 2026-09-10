"""Vehicle-barrier crash (3D) benchmark -- PRIVATE PLACEHOLDER (ADR-0058).

See ``benchmark.py``'s module docstring for the full rationale: this
benchmark's data is private and not distributable, so it is registered (for
the shared training pipeline) but excluded from the public portfolio via
``registry._UNLISTED``, and carries no results.
"""

from ..registry import BenchmarkSpec
from ..results import BaselineResult
from .benchmark import AUX_FIELD, KINEMATIC_TYPES, QOIS, TEST, TRAIN, VAL
from .card import CARD

__all__ = [
    "AUX_FIELD",
    "CARD",
    "KINEMATIC_TYPES",
    "QOIS",
    "SPEC",
    "TEST",
    "TRAIN",
    "VAL",
]

#: Permanently empty (ADR-0058): the data is private, so no run on it is ever
#: blessed or listed as an official/provisional baseline.
RESULTS: tuple[BaselineResult, ...] = ()

# TODO(ADR-0058): once data_generation/lsdyna/VehicleBarrierCrash/ exists and
# the real per-case layer-thickness metadata convention is known, wire up a
# _layer_thickness(case_id) -> tuple[float, float, float, float] extractor
# here (0.0 = no layer at that position) and pass it as SPEC.loading_scalars,
# with TransolverConfig.loading_features=4 on the run config side. Left
# unset (None) for now rather than guessing at a case-id format that doesn't
# exist yet.

SPEC = BenchmarkSpec(
    card=CARD,
    results=RESULTS,
    splits={
        "train": tuple(TRAIN),
        "val": tuple(VAL),
        "test": tuple(TEST),
    },
    eval_splits=("val", "test"),
    aux_field=AUX_FIELD,
    qois=dict(QOIS),
    boundary_feature_fn=None,
    dataset_id="VehicleBarrierCrash-3D",
    kinematic_types=KINEMATIC_TYPES,
    mesh_transform=None,
    scripted_types=KINEMATIC_TYPES,
    loading_scalars=None,
)
