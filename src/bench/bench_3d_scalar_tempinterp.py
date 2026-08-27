# ==============================================================================
# felix benchmark: 3d scalar field with temporal interpolation
# ==============================================================================

from bench.bench_common import run_benchmark_case

CASE_NAME = "bench_3d_scalar_tempinterp"


def main() -> None:
    run_benchmark_case(
        case_name=CASE_NAME,
        spatial_dims=3,
        field_kind="scalar",
        use_temp_interp=True,
    )


if __name__ == "__main__":
    main()
