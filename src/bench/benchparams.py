# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from pathlib import Path

# Sizing and workloads
SENSORS_NUM: int = 100
SIM_TIMES_NUM: int = 50
SAMPLE_TIMES_NUM: int = 200

# Synthetic mesh divisions (elements along each axis)
MESH_DIVS_2D: tuple[int, int] = (20, 20)
MESH_DIVS_3D: tuple[int, int, int] = (10, 10, 10)

# Physical domain bounding box: [0, LENGTH_X] x [0, LENGTH_Y] (x [0, LENGTH_Z])
DOMAIN_LENGTH_X: float = 100.0
DOMAIN_LENGTH_Y: float = 50.0
DOMAIN_LENGTH_Z: float = 20.0

# Timing and repetition parameters
CALC_MEAS_CALLS: int = 200
RUNS_PER_CASE: int = 30
WARMUP_RUNS: int = 2

# Benchmark output directory
BENCH_OUTPUT_DIR: Path = Path.cwd() / "out" / "bench"
