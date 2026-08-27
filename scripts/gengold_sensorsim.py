"""Generate reviewed sensor-simulation regression data for Felix."""

import argparse
from pathlib import Path

import numpy as np
import pyvale.verif.pointsensscalar as pointsensscalar
import pyvale.verif.pointsenstensor as pointsenstensor
import pyvale.verif.pointsensvector as pointsensvector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = PROJECT_ROOT / "src" / "felix" / "pytests" / "gold"

SENSOR_SUITES = (
    pointsensscalar.sens_arrays_2d_dict,
    pointsensscalar.sens_arrays_3d_dict,
    pointsensvector.sens_arrays_2d_dict,
    pointsensvector.sens_arrays_3d_dict,
    pointsenstensor.sens_arrays_2d_dict,
    pointsenstensor.sens_arrays_3d_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the generated arrays to the committed gold directory",
    )
    parser.add_argument(
        "--case",
        help="generate only the case with this exact tag",
    )
    args = parser.parse_args()
    if not args.write:
        parser.error("choose --write to refresh committed gold data")
    return args


def main() -> None:
    args = parse_args()
    matched = False

    for load_suite in SENSOR_SUITES:
        for tag, sensors in load_suite().items():
            if args.case is not None and tag != args.case:
                continue
            matched = True
            output_path = GOLD_DIR / f"{tag}.npy"
            np.save(output_path, sensors.sim_measurements())
            print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}")

    if args.case is not None and not matched:
        raise ValueError(f"Unknown sensor-simulation case: {args.case}")


if __name__ == "__main__":
    main()
