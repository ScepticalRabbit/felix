# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from bench.bench_common import run_benchmark_case

CASE_NAME = "bench_3d_vector_tempinterp"


def main() -> None:
    run_benchmark_case(
        case_name=CASE_NAME,
        spatial_dims=3,
        field_kind="vector",
        use_temp_interp=True,
    )


if __name__ == "__main__":
    main()
