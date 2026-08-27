# --------------------------------------------------------------------------
# Felix: A High Performance Sensor Simulation Core
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
"""Example 0: Pass a pyvale SensorData object down to Zig for printing.

This example:
1. Creates a minimal pyvale.sensorsim.SensorData object.
2. Extracts the underlying NumPy arrays.
3. Passes them through the Cython wrapper to the Felix Zig core.
4. The Zig core prints the data to stderr, proving the call path works.

Run from the project root after building with:
    pip install -e .
    python src/pyexs/ex0_print_sensordata.py
"""
import numpy as np
import pyvale.sensorsim as ss
from felix.cython import felix as fc


def main() -> None:
    # ------------------------------------------------------------------
    # Build a minimal SensorData with 3 sensors
    # ------------------------------------------------------------------
    positions = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.025, 0.050, 0.000],
            [0.050, 0.100, 0.000],
        ],
        dtype=np.float64,
    )

    sample_times = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4],
        dtype=np.float64,
    )

    spatial_dims = np.array(
        [0.005, 0.005, 0.001],
        dtype=np.float64,
    )

    sensor_data = ss.SensorData(
        positions=positions,
        sample_times=sample_times,
        spatial_dims=spatial_dims,
    )

    print("Python: created SensorData")
    print(f"  positions shape : {sensor_data.positions.shape}")
    print(f"  sample_times    : {sensor_data.sample_times}")
    print(f"  spatial_dims    : {sensor_data.spatial_dims}")
    print()
    print("Python: calling felix.cython.felix.print_sensor_data ...")
    print()

    # ------------------------------------------------------------------
    # Forward the arrays to the Zig core via the Cython wrapper
    # ------------------------------------------------------------------
    fc.print_sensor_data(
        sensor_data.positions,
        sensor_data.sample_times,
        sensor_data.spatial_dims,
    )

    print("Python: Zig call returned successfully.")


if __name__ == "__main__":
    main()
