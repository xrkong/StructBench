"""Adapter: collider's downsampled crash HDF5 -> canonical Case (ADR-0058).

collider (a companion research repo, not part of this package) already
solves the hard part of ingesting a raw LS-DYNA vehicle-vs-barrier crash deck
-- joining d3plot part indices to k-file PIDs by title (unreliable via
d3plot's ``part_titles_ids`` alone, per ``core.io.lsdyna._part_id_lookup``'s
docstring), region-aware downsampling to ~100k nodes, and scattering
element-level effective plastic strain onto nodes. Its output is one HDF5
per case with a fixed schema (see ``read_vehicle_barrier_crash``'s
docstring). This adapter converts THAT already-processed schema into a
canonical :class:`~structbench.core.schema.Case` -- it does not touch the
raw d3plot/k-file at all, so ``core.io.lsdyna``'s PID-join question never
arises here.

Mirrors ``core.io.meshgraphnets``'s split into a pure ``read_*`` (I/O) and a
pure ``build_*`` (assembly, testable without a real file) function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..schema import Case, ElementBlock, Material, Metadata, Nodes, Provenance, Response
from ..validation import validate
from .lsdyna import unit_factors

#: region_id -> canonical node_type (ADR-0058). collider's 6-way sampling
#: region (0=force_keep, 1=barrier_fine, 2=barrier_coarse, 3=veh_contact,
#: 4=veh_near, 5=veh_far -- see collider's dataset/constants.py
#: REGION_ID_LEGEND) collapses to a 2-way vehicle/barrier split for the
#: model's node-type one-hot: 0=vehicle (regions 0,3,4,5 -- force_keep nodes
#: are accelerometer/dummy points rigidly attached to the vehicle cabin, not
#: barrier structure), 1=barrier (regions 1,2). Neither is "kinematic": this
#: benchmark has no externally-prescribed boundary (confirmed against real
#: data, 2026-09-09) -- both the vehicle body and the barrier deform freely
#: under the crash physics, so BenchmarkSpec.kinematic_types is () and this
#: node_type exists purely as an input-conditioning feature, not a
#: loss/rollout exclusion mask.
_REGION_TO_NODE_TYPE = {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0}
NODE_TYPE_SIZE = 2


def region_id_to_node_type(region_id: np.ndarray) -> np.ndarray:
    """Map collider's 6-way region_id to the 2-way vehicle(0)/barrier(1) node_type."""
    lut = np.zeros(max(_REGION_TO_NODE_TYPE) + 1, dtype=np.int64)
    for region, node_type in _REGION_TO_NODE_TYPE.items():
        lut[region] = node_type
    return lut[np.asarray(region_id, dtype=np.int64)]


def read_vehicle_barrier_crash(path: str | Path) -> dict[str, Any]:
    """Read one collider-format downsampled crash HDF5 into plain arrays.

    Parameters
    ----------
    path:
        Path to a ``<case>.h5`` file written by collider's
        ``dataset/build_dataset.py`` (schema: ``/metadata/{sampled_node_ids,
        ref_positions, region_id, node_part_id, node_part_name,
        node_mat_type_name, shell_cells, solid_cells, beam_cells, ...}`` +
        ``/states/{times, positions, eff_plastic_strain, node_alive}``, attrs
        including ``units`` e.g. ``"mm, ton, s, MPa"``).

    Returns
    -------
    dict[str, Any]
        ``ref_positions`` (N,3), ``region_id`` (N,), ``shell_cells`` (E,4),
        ``node_mat_type_name`` (N,) bytes, ``positions`` (T,N,3),
        ``eff_plastic_strain`` (T,N), ``times`` (T,), ``units`` str.
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        mg = f["metadata"]
        sg = f["states"]
        return {
            "ref_positions": mg["ref_positions"][...],
            "region_id": mg["region_id"][...],
            "shell_cells": mg["shell_cells"][...],
            "node_mat_type_name": mg["node_mat_type_name"][...],
            "positions": sg["positions"][...],
            "eff_plastic_strain": sg["eff_plastic_strain"][...],
            "times": sg["times"][...],
            "units": mg.attrs["units"],
        }


def build_vehicle_barrier_crash_case(
    arrays: dict[str, Any],
    *,
    source_units: str,
    case_id: str,
    dataset_id: str | None = None,
) -> Case:
    """Assemble one collider-format crash trajectory into a validated 3D Case.

    Narrows to a SINGLE element block (``shell`` -- collider's shell mesh
    dominates by element count over its solid/beam blocks, and
    ``datasets.canonical._load_mesh_trajectory`` requires exactly one
    element type). Nodes covered only by dropped solid/beam elements are
    unaffected: ``case.nodes`` unconditionally keeps every node regardless of
    which element block survives (ADR-0058, confirmed against
    ``datasets/canonical.py``).

    Parameters
    ----------
    arrays:
        As returned by :func:`read_vehicle_barrier_crash`.
    source_units:
        ``"mass-length-time"`` token, e.g. ``"t-mm-s"`` (collider's own
        ``"mm, ton, s, MPa"`` convention).
    case_id:
        Unique identifier for the assembled case (collider's case name,
        e.g. ``"New_Road_Barrier_concrete_W_beam_four_layer_03_1_5mm"``).
    dataset_id:
        Optional identifier for the source dataset this case belongs to.

    Returns
    -------
    Case
        A validated case: ``nodes.coords`` / ``reference_coords`` are the
        rest-frame positions (SI), ``response.node["displacement"]`` is the
        delta from rest per frame, ``response.node["effective_plastic_strain"]``
        is collider's already node-scattered field, and ``nodes.node_type``
        is the 2-way vehicle/barrier split (see
        :data:`_REGION_TO_NODE_TYPE`'s docstring for why nothing is
        kinematic here).
    """
    f = unit_factors(source_units)
    ref = arrays["ref_positions"].astype(np.float64) * f["length"]  # (N, 3)
    pos = arrays["positions"].astype(np.float64) * f["length"]  # (T, N, 3)
    disp = pos - ref[None]  # (T, N, 3), delta-from-rest
    n_nodes = ref.shape[0]

    nodes = Nodes(
        coords=ref,
        node_id=np.arange(n_nodes, dtype=np.int64),
        node_type=region_id_to_node_type(arrays["region_id"]),
        reference_coords=ref.copy(),
    )
    shell_cells = arrays["shell_cells"].astype(np.int64)
    elements = {
        "shell": ElementBlock(
            connectivity=shell_cells,
            element_id=np.arange(shell_cells.shape[0], dtype=np.int64),
            part_id=np.zeros(shell_cells.shape[0], dtype=np.int64),
        )
    }
    eps = arrays["eff_plastic_strain"].astype(np.float64)[..., None]  # (T, N, 1)
    response = Response(
        time=arrays["times"].astype(np.float64) * f["time"],
        node={
            "displacement": disp.astype(np.float32),
            "effective_plastic_strain": eps.astype(np.float32),
        },
    )
    metadata = Metadata(
        case_id=case_id,
        dimension=3,
        source_units=source_units,
        dataset_id=dataset_id,
        provenance=Provenance("LS-DYNA", "unknown", "unknown"),
    )
    mat_names = sorted(
        {n.decode(errors="replace") for n in arrays["node_mat_type_name"]}
    )
    materials = [
        Material(
            material_id=i,
            source_model=f"MAT_{name}",
            source_params={
                "note": (
                    "per-node material properties from collider's downsampled "
                    "HDF5 (node_mat_rho/E/nu/sigy), not carried through here"
                )
            },
            canonical_model=None,
        )
        for i, name in enumerate(mat_names)
    ]
    case = Case(
        metadata=metadata,
        nodes=nodes,
        elements=elements,
        materials=materials,
        response=response,
    )
    validate(case)
    return case
