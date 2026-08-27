# ==============================================================================
# felix benchmark: 2d tensor field without temporal interpolation
# ==============================================================================

from bench.bench_common import run_benchmark_case

CASE_NAME = "bench_2d_tensor_nointerp"


def main() -> None:
    run_benchmark_case(
        case_name=CASE_NAME,
        spatial_dims=2,
        field_kind="tensor",
        use_temp_interp=False,
    )


if __name__ == "__main__":
    main()
