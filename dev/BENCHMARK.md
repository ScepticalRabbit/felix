# Benchmarking Guide

This guide describes how to run, configure, and interpret performance benchmarks comparing the Felix Zig-native core against the Python/NumPy Pyvale sensor simulation baseline.

---

## 1. Overview & Benchmark Suite Structure

The benchmarking suite lives in `src/bench/` and evaluates raw sensor simulation truth sampling performance across all standard combinations of:
- **Spatial Dimensions**: 2D (QUAD4) and 3D (HEX8)
- **Field Kinds**: Scalar, Vector (with rotations), Tensor (with coordinate transformations)
- **Temporal Modes**: Without temporal interpolation (sampling at exact simulation time steps) and with temporal interpolation (sampling at arbitrary continuous query times)

### Benchmark Cases (12 Total):

| Case File | Dimension | Field Kind | Temporal Interp |
|---|---|---|---|
| `bench_2d_scalar_nointerp.py` | 2D | Scalar | No |
| `bench_2d_scalar_tempinterp.py` | 2D | Scalar | Yes |
| `bench_2d_vector_nointerp.py` | 2D | Vector | No |
| `bench_2d_vector_tempinterp.py` | 2D | Vector | Yes |
| `bench_2d_tensor_nointerp.py` | 2D | Tensor | No |
| `bench_2d_tensor_tempinterp.py` | 2D | Tensor | Yes |
| `bench_3d_scalar_nointerp.py` | 3D | Scalar | No |
| `bench_3d_scalar_tempinterp.py` | 3D | Scalar | Yes |
| `bench_3d_vector_nointerp.py` | 3D | Vector | No |
| `bench_3d_vector_tempinterp.py` | 3D | Vector | Yes |
| `bench_3d_tensor_nointerp.py` | 3D | Tensor | No |
| `bench_3d_tensor_tempinterp.py` | 3D | Tensor | Yes |

---

## 2. Configuration Parameters (`src/bench/benchparams.py`)

All benchmark workloads and repetitions are configured in `src/bench/benchparams.py`:

```python
# Workload Sizing
SENSORS_NUM: int = 100         # Number of virtual point sensors
SIM_TIMES_NUM: int = 50        # Number of simulation time steps
SAMPLE_TIMES_NUM: int = 200    # Number of query sample times

# Synthetic Mesh Resolution
MESH_DIVS_2D: tuple[int, int] = (20, 20)        # 400 elements, 441 nodes
MESH_DIVS_3D: tuple[int, int, int] = (10, 10, 10) # 1000 elements, 1331 nodes

# Repetitions & Timing
CALC_MEAS_CALLS: int = 1000    # Number of consecutive calc_truth calls per run
RUNS_PER_CASE: int = 30        # Number of independent benchmark runs
WARMUP_RUNS: int = 2           # Number of untimed warmup runs before measurement

# Destination
BENCH_OUTPUT_DIR: Path = Path.cwd() / "out" / "bench"
```

---

## 3. Running Benchmarks

### Running the Entire Benchmark Suite:

Run from the repository root using the active Python virtual environment:

```bash
python -m bench.run_all_bench
```

This sequentially executes all 12 benchmark cases, writes per-case CSV logs into `out/bench/<case_name>/`, and automatically runs the statistical summary generator.

### Running a Specific Benchmark Case:

To run a single benchmark case:

```bash
python -m bench.bench_2d_scalar_nointerp
python -m bench.bench_3d_vector_tempinterp
```

---

## 4. Benchmark Outputs and Results Processing

### 1. Per-Case Raw CSV Outputs
Each benchmark execution writes a timestamped CSV file to:
`out/bench/<case_name>/<case_name>_YYYYMMDD_HHMMSS.csv`

Each row records an individual run containing:
- `run_idx`: Run repetition index (0 to `RUNS_PER_CASE - 1`)
- `pyvale_wall_s`: Total wall-clock time in seconds for Pyvale across `CALC_MEAS_CALLS`
- `felix_wall_s`: Total wall-clock time in seconds for Felix across `CALC_MEAS_CALLS`
- `speedup`: `pyvale_wall_s / felix_wall_s`

### 2. Generating & Interpreting Aggregate Statistics

To re-aggregate and view the latest benchmark results at any time:

```bash
python -m bench.bench_stats
```

This finds the latest timestamped CSV for each case in `out/bench/`, computes statistical metrics (mean, median, standard deviation, min, max, speedup ratios, and evaluations per second), saves an aggregate CSV to `out/bench/bench_stats_YYYYMMDD_HHMMSS.csv`, and prints a formatted summary table:

```
==========================================================================================
 FELIX vs PYVALE BENCHMARK SUMMARY (20260827_171500)
==========================================================================================
Case Name                        | Dims | Field  | Interp | Pyvale (s)  | Felix (s)  | Speedup 
------------------------------------------------------------------------------------------
bench_2d_scalar_nointerp         | 2D   | scalar | No     | 0.8234      | 0.1381     | 5.96x   
bench_2d_scalar_tempinterp       | 2D   | scalar | Yes    | 0.8712      | 0.2945     | 2.96x   
bench_2d_vector_nointerp         | 2D   | vector | No     | 1.4180      | 0.4087     | 3.47x   
...
==========================================================================================
```

### Metrics Interpretation:
- **Speedup**: Computed as `pyvale_median_s / felix_median_s`. A speedup of `5.96x` means Felix completes the batch simulation nearly 6 times faster than Pyvale.
- **Latency Per Call**: Can be obtained by dividing `felix_median_s` by `CALC_MEAS_CALLS` (e.g. `0.1381 s / 1000 = 138.1 µs/call` for 100 sensors across 50 time steps).
- **Throughput**: Expressed in total evaluations per second (`sensors_num * sample_times_num * calc_meas_calls / felix_median_s`).
