# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import time
from datetime import datetime

from bench.bench_stats import generate_and_print_summary
import bench.bench_2d_scalar_nointerp as b_2d_s_no
import bench.bench_2d_scalar_tempinterp as b_2d_s_ti
import bench.bench_2d_vector_nointerp as b_2d_v_no
import bench.bench_2d_vector_tempinterp as b_2d_v_ti
import bench.bench_2d_tensor_nointerp as b_2d_t_no
import bench.bench_2d_tensor_tempinterp as b_2d_t_ti

import bench.bench_3d_scalar_nointerp as b_3d_s_no
import bench.bench_3d_scalar_tempinterp as b_3d_s_ti
import bench.bench_3d_vector_nointerp as b_3d_v_no
import bench.bench_3d_vector_tempinterp as b_3d_v_ti
import bench.bench_3d_tensor_nointerp as b_3d_t_no
import bench.bench_3d_tensor_tempinterp as b_3d_t_ti

ALL_BENCHMARKS = [
    ("2D Scalar NoInterp", b_2d_s_no.main),
    ("2D Scalar TempInterp", b_2d_s_ti.main),
    ("2D Vector NoInterp", b_2d_v_no.main),
    ("2D Vector TempInterp", b_2d_v_ti.main),
    ("2D Tensor NoInterp", b_2d_t_no.main),
    ("2D Tensor TempInterp", b_2d_t_ti.main),
    ("3D Scalar NoInterp", b_3d_s_no.main),
    ("3D Scalar TempInterp", b_3d_s_ti.main),
    ("3D Vector NoInterp", b_3d_v_no.main),
    ("3D Vector TempInterp", b_3d_v_ti.main),
    ("3D Tensor NoInterp", b_3d_t_no.main),
    ("3D Tensor TempInterp", b_3d_t_ti.main),
]


def main() -> None:
    start_time = time.perf_counter()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print(f" STARTING FELIX-ONLY BENCHMARK SUITE ({now_str})")
    print("=" * 80)

    for idx, (label, bench_func) in enumerate(ALL_BENCHMARKS, start=1):
        print(f"\n[{idx}/{len(ALL_BENCHMARKS)}] Running {label}...")
        bench_func(mode="felix_only")

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 80)
    print(f" FELIX BENCHMARKS COMPLETED IN {total_time:.2f} SECONDS")
    print("=" * 80)

    # Generate and print statistical summary
    generate_and_print_summary()


if __name__ == "__main__":
    main()
