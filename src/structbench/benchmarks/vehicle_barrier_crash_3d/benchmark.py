"""Vehicle-barrier crash (3D) task facts: split, aux field, kinematics, QoIs.

PRIVATE (ADR-0058): unlike every other StructBench benchmark's canonical
dataset (maintainer-held on OneDrive and shared on request, ADR-0040), the
underlying LS-DYNA vehicle-vs-barrier crash data is not distributable at all
-- this module exists so the shared structbench-train pipeline can train
against it, not so it can be published. Deliberately excluded from the
public benchmark portfolio (registry's ``_UNLISTED``) and carries no blessed
or provisional results.

Real data (2026-09-09): 10 cases from the "New_Road_Barrier" family (varying
barrier layer configuration), converted from collider's own downsampled
HDF5 via ``data_generation/lsdyna/VehicleBarrierCrash3D/convert.py``. The
"T_lok_F_shape_barrier" family (varying impact speed, a different
conditioning axis) is a separate, not-yet-ingested case set -- see that
script's module docstring. Every case: 100,046 nodes, 50 frames.

Confirmed against real data: this benchmark has no externally-prescribed
boundary (unlike Taylor's wall, DeformingPlate's actuator, or notch-beam's
pin/support) -- both the vehicle body and the barrier deform freely under
the crash physics, so ``KINEMATIC_TYPES`` is empty. See
``structbench.core.io.vehicle_barrier_crash``'s module docstring for the
node_type derivation.
"""

from __future__ import annotations

from ...eval import QoiFn, peak_nodal_aux, terminal_peak_displacement

#: Frozen split (ADR-0058). Only 10 total cases -- this split is a
#: provisional, easily-revised starting point, not a considered protocol
#: decision (no ground-truth timeline analysis has been run; see
#: scratch/2026-09-09-vbc3d-train-tasks.md).
TRAIN: list[str] = [
    "New_Road_Barrier_concrete_W_beam",
    "New_Road_Barrier_concrete_W_beam_four_layer",
    "New_Road_Barrier_concrete_W_beam_four_layer_03_1_5mm",
    "New_Road_Barrier_concrete_W_beam_four_layer_03_2mm",
    "New_Road_Barrier_concrete_W_beam_four_layer_06_2mm",
    "New_Road_Barrier_W_beam_two_layer",
    "New_Road_Barrier_W_beam_three_layer",
]
VAL: list[str] = [
    "New_Road_Barrier_concrete_W_beam_four_layer_05_1_5mm",
]
TEST: list[str] = [
    "New_Road_Barrier_concrete_W_beam_four_layer_07_2mm",
    "New_Road_Barrier_10_14_FD_0_3_0_1_LR_NoThic",
]

#: Auxiliary per-node target: effective plastic strain, scattered from
#: element response onto nodes at ingestion time (the natural damage-adjacent
#: quantity for a crash/impact structure; distinct from the notch-beam
#: benchmarks' max_principal_strain, ADR-0029).
AUX_FIELD = "effective_plastic_strain"

#: No externally-prescribed boundary in this data (confirmed 2026-09-09,
#: see module docstring) -- nothing is kinematic. TransolverConfig's
#: kinematic-BC input channel (time-conditioned scheme) is therefore inert
#: here; conditioning comes entirely through the scalar loading_features
#: channel (barrier layer thickness) instead.
KINEMATIC_TYPES: tuple[int, ...] = ()

QOIS: dict[str, QoiFn] = {
    "peak_plastic_strain": peak_nodal_aux(exclude_types=KINEMATIC_TYPES),
    "terminal_peak_deflection": terminal_peak_displacement(
        exclude_types=KINEMATIC_TYPES
    ),
}
