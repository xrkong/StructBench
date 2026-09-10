"""Convert collider's downsampled vehicle-barrier-crash HDF5 to canonical HDF5.

Per-dataset glue (ADR-0016 §6/ADR-0058). Source data is NOT raw d3plot/deck --
it's collider's own already-processed HDF5 (region-aware downsampled to
~100k nodes, part-id joined by title, effective plastic strain already
scattered from elements onto nodes). All the actual format-conversion logic
(unit conversion, node_type derivation, narrowing to a single element block)
lives in ``structbench.core.io.vehicle_barrier_crash``; this script only
knows this dataset's specifics: where the files live and what case ids to
convert.

This first pass covers only the "New_Road_Barrier" family (varying barrier
layer configuration) -- NOT the "T_lok_F_shape_barrier" family (varying
impact speed), which uses a different scalar-conditioning channel
(impact_velocity_feature, not loading_features) and is deliberately deferred
to keep the two conditioning axes from mixing in one benchmark.

Usage (from the repo root):
    python data_generation/lsdyna/VehicleBarrierCrash3D/convert.py \\
        --src /raid/proj_iim1/xrkong/h5_fps_no_wheel --out <data_root>
    python data_generation/lsdyna/VehicleBarrierCrash3D/convert.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from structbench.core import write_case
from structbench.core.io.vehicle_barrier_crash import (
    build_vehicle_barrier_crash_case,
    read_vehicle_barrier_crash,
)

DATASET_ID = "VehicleBarrierCrash3D"
SOURCE_UNITS = "t-mm-s"  # collider's own convention: "mm, ton, s, MPa"

#: New_Road_Barrier family only (layer-thickness conditioning axis; see
#: module docstring for why T_lok_F_shape_barrier -- speed conditioning -- is
#: excluded from this first pass).
CASE_IDS: tuple[str, ...] = (
    "New_Road_Barrier_10_14_FD_0_3_0_1_LR_NoThic",
    "New_Road_Barrier_W_beam_three_layer",
    "New_Road_Barrier_W_beam_two_layer",
    "New_Road_Barrier_concrete_W_beam",
    "New_Road_Barrier_concrete_W_beam_four_layer",
    "New_Road_Barrier_concrete_W_beam_four_layer_03_1_5mm",
    "New_Road_Barrier_concrete_W_beam_four_layer_03_2mm",
    "New_Road_Barrier_concrete_W_beam_four_layer_05_1_5mm",
    "New_Road_Barrier_concrete_W_beam_four_layer_06_2mm",
    "New_Road_Barrier_concrete_W_beam_four_layer_07_2mm",
)

_LOG = logging.getLogger("convert")


def convert_one(src_dir: Path, out_dir: Path, case_id: str) -> Path:
    """Convert one case; returns the written canonical file's path."""
    src = src_dir / f"{case_id}.h5"
    arrays = read_vehicle_barrier_crash(src)
    case = build_vehicle_barrier_crash_case(
        arrays, source_units=SOURCE_UNITS, case_id=case_id, dataset_id=DATASET_ID
    )
    out_path = out_dir / f"{case_id}.h5"
    write_case(case, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/raid/proj_iim1/xrkong/h5_fps_no_wheel"),
        help="Directory of collider's <case_id>.h5 files.",
    )
    parser.add_argument(
        "--out", type=Path, required=False, help="Canonical output dir."
    )
    parser.add_argument("--case", type=str, default=None, help="Convert one case only.")
    parser.add_argument(
        "--dry-run", action="store_true", help="List cases without reading any file."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    case_ids = (args.case,) if args.case else CASE_IDS
    if args.dry_run:
        for cid in case_ids:
            print(cid)
        return 0

    if args.out is None:
        parser.error("--out is required unless --dry-run")
    args.out.mkdir(parents=True, exist_ok=True)

    for cid in case_ids:
        _LOG.info("converting %s", cid)
        out_path = convert_one(args.src, args.out, cid)
        _LOG.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
