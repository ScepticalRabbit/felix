# Benchmarking Guide

This guide describes how to run, configure, and interpret performance benchmarks for Felix, both in standalone mode (**Felix Only**) and in comparative mode (**Felix vs. Pyvale**).

---

## 1. Overview & Suite Structure

The benchmarking suite lives in `src/bench/` and evaluates sensor simulation truth sampling throughput and latency across standard combinations of:
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

Workload sizing and repetitions are configured in `src/bench/benchparams.py`:

```python
# Workload Sizing
SENSORS_NUM: int = 128         # Number of virtual point sensors
SIM_TIMES_NUM: int = 50        # Number of simulation time steps
SAMPLE_TIMES_NUM: int = 128    # Number of query sample times

# Synthetic Mesh Resolution
MESH_DIVS_2D: tuple[int, int] = (20, 20)          # 400 elements, 441 nodes
MESH_DIVS_3D: tuple[int, int, int] = (10, 10, 10) # 1000 elements, 1331 nodes

# Repetitions & Timing
CALC_MEAS_CALLS: int = 128     # Consecutive calc_truth calls per run
RUNS_PER_CASE: int = 32        # Independent benchmark runs
WARMUP_RUNS: int = 2           # Untimed warmup runs before measurement

# Output Directory
BENCH_OUTPUT_DIR: Path = Path.cwd() / "out" / "bench"
```

---

## 3. Running Benchmarks

### Option A: Standalone Felix Benchmark (Felix Only)

Runs the complete 12-case benchmark suite measuring only Felix (eliminating all Pyvale instantiation and evaluation overhead):

```bash
.venv/bin/python src/bench/bench_felix_only.py
# or
.venv/bin/python src/bench/run_all_bench.py --felix-only
```

### Option B: Comparative Benchmark (Felix vs. Pyvale)

Runs all 12 cases comparing Felix against the Pyvale baseline, including numerical parity verification (`assert_allclose`) and speedup calculation:

```bash
.venv/bin/python src/bench/bench_comp_pyvale.py
# or
.venv/bin/python src/bench/run_all_bench.py
```

### Option C: Running a Specific Case

Each individual benchmark script can be executed standalone:

```bash
# Felix Only:
.venv/bin/python -c "from bench.bench_2d_scalar_nointerp import main; main(mode='felix_only')"

# Comparative:
.venv/bin/python -c "from bench.bench_3d_tensor_tempinterp import main; main(mode='comp_pyvale')"
```

---

## 4. Benchmark Outputs and Statistical Processing

### 1. Raw Output CSVs
Each run writes a timestamped CSV to `out/bench/<case_name>/<case_name>_YYYYMMDD_HHMMSS.csv` recording:
- `felix_wall_s`: Total wall-clock time in seconds for Felix across `CALC_MEAS_CALLS`
- `felix_s_per_call`: Latency per evaluation
- `pyvale_wall_s` and `speedup`: Recorded when running in comparative mode

### 2. Aggregate Summary
To regenerate the latest aggregate benchmark summary at any time:

```bash
.venv/bin/python src/bench/bench_stats.py
```

This parses all latest case CSVs, exports an aggregate summary to `out/bench/bench_stats_YYYYMMDD_HHMMSS.csv`, and prints a formatted terminal summary.
