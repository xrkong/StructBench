from __future__ import annotations

import numpy as np

from structbench.core import validate
from structbench.core.io.vehicle_barrier_crash import (
    NODE_TYPE_SIZE,
    build_vehicle_barrier_crash_case,
    region_id_to_node_type,
)


def _synthetic_arrays(n_nodes=8, n_cells=2, T=4):
    rng = np.random.default_rng(0)
    ref = rng.random((n_nodes, 3)).astype(np.float32) * 1000.0  # mm
    pos = np.stack([ref + i * 10.0 for i in range(T)]).astype(np.float32)  # (T, N, 3)
    region_id = np.array([0, 1, 2, 3, 4, 5, 3, 1], dtype=np.int64)[:n_nodes]
    return {
        "ref_positions": ref,
        "region_id": region_id,
        "shell_cells": rng.integers(0, n_nodes, (n_cells, 4)).astype(np.int32),
        "node_mat_type_name": np.array(
            [b"RIGID", b"CONCRETE_DAMAGE_REL3"] * (n_nodes // 2 + 1)
        )[:n_nodes],
        "positions": pos,
        "eff_plastic_strain": (rng.random((T, n_nodes)) * 2.0).astype(np.float32),
        "times": np.arange(T, dtype=np.float64) * 4e-5,  # frame_stride-scale seconds
    }


def test_region_id_to_node_type_maps_vehicle_and_barrier():
    # 0=force_keep, 1=barrier_fine, 2=barrier_coarse, 3=veh_contact,
    # 4=veh_near, 5=veh_far (collider's REGION_ID_LEGEND)
    region_id = np.array([0, 1, 2, 3, 4, 5])
    node_type = region_id_to_node_type(region_id)
    np.testing.assert_array_equal(node_type, [0, 1, 1, 0, 0, 0])
    assert NODE_TYPE_SIZE == 2


def test_build_case_maps_fields_and_validates():
    a = _synthetic_arrays()
    case = build_vehicle_barrier_crash_case(
        a, source_units="t-mm-s", case_id="vbc-000", dataset_id="VehicleBarrierCrash3D"
    )
    validate(case)  # must not raise
    assert case.metadata.dimension == 3
    assert case.metadata.units_convention == "SI"
    assert set(case.elements) == {"shell"}  # narrowed to a single element block

    # units: t-mm-s length factor is 1e-3 (mm -> m); coords == ref_positions in SI
    np.testing.assert_allclose(case.nodes.coords, a["ref_positions"] * 1e-3, rtol=1e-5)
    np.testing.assert_allclose(
        case.nodes.reference_coords, a["ref_positions"] * 1e-3, rtol=1e-5
    )
    np.testing.assert_array_equal(
        case.nodes.node_type, region_id_to_node_type(a["region_id"])
    )

    # displacement is delta-from-rest (frame 0 == rest, later frames grow)
    np.testing.assert_allclose(
        case.response.node["displacement"][0], np.zeros((8, 3)), atol=1e-4
    )
    disp2 = case.response.node["displacement"][2]
    assert float(np.abs(disp2).max()) > 0

    # effective_plastic_strain carries through UNSCALED (dimensionless,
    # ADR-0058 -- the same bug class the stress_scale mesh-path fix guards)
    np.testing.assert_allclose(
        case.response.node["effective_plastic_strain"][:, :, 0],
        a["eff_plastic_strain"],
        rtol=1e-5,
    )

    assert len(case.materials) >= 1
    assert any("RIGID" in m.source_model for m in case.materials)
