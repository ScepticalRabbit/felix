# ==============================================================================
# felix: benchmark statistical analysis and aggregation
# ==============================================================================

import csv
from datetime import datetime
from pathlib import Path
import numpy as np

from bench.benchparams import BENCH_OUTPUT_DIR


def find_latest_case_csvs(bench_dir: Path) -> dict[str, Path]:
    """Finds the most recent CSV result file for each benchmark case."""
    if not bench_dir.exists():
        return {}

    latest_files: dict[str, tuple[str, Path]] = {}

    for case_dir in bench_dir.iterdir():
        if not case_dir.is_dir():
            continue

        case_name = case_dir.name
        csv_files = list(case_dir.glob(f"{case_name}_*.csv"))
        if not csv_files:
            continue

        # Sort by filename which contains timestamp YYYYMMDD_HHMMSS
        csv_files.sort(key=lambda p: p.stem, reverse=True)
        latest_file = csv_files[0]
        latest_files[case_name] = (latest_file.stem, latest_file)

    return {k: v[1] for k, v in sorted(latest_files.items())}


def analyze_case_csv(csv_path: Path) -> dict:
    """Reads a benchmark run CSV and computes summary statistics."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    py_times = np.array([float(r["pyvale_wall_s"]) for r in rows])
    fx_times = np.array([float(r["felix_wall_s"]) for r in rows])
    speedups = np.array([float(r["speedup"]) for r in rows])

    first_row = rows[0]
    num_sensors = int(first_row["sensors_num"])
    num_samples = int(first_row["sample_times_num"])
    num_calls = int(first_row["calc_meas_calls"])
    total_evals = num_sensors * num_samples * num_calls

    py_med = float(np.median(py_times))
    fx_med = float(np.median(fx_times))
    fx_mean = float(np.mean(fx_times))

    stats = {
        "case_name": first_row["case_name"],
        "spatial_dims": int(first_row["spatial_dims"]),
        "field_kind": first_row["field_kind"],
        "use_temp_interp": first_row["use_temp_interp"],
        "sensors_num": num_sensors,
        "sample_times_num": num_samples,
        "calc_meas_calls": num_calls,
        "runs_count": len(rows),
        "pyvale_median_s": py_med,
        "pyvale_mean_s": float(np.mean(py_times)),
        "pyvale_std_s": float(np.std(py_times)),
        "pyvale_min_s": float(np.min(py_times)),
        "pyvale_max_s": float(np.max(py_times)),
        "felix_median_s": fx_med,
        "felix_mean_s": fx_mean,
        "felix_std_s": float(np.std(fx_times)),
        "felix_min_s": float(np.min(fx_times)),
        "felix_max_s": float(np.max(fx_times)),
        "speedup_median": py_med / fx_med if fx_med > 0 else 0.0,
        "speedup_mean": float(np.mean(speedups)),
        "felix_evals_per_sec": total_evals / fx_med if fx_med > 0 else 0.0,
        "csv_source": str(csv_path),
    }
    return stats


def generate_and_print_summary(
    bench_dir: Path = BENCH_OUTPUT_DIR,
) -> Path | None:
    """Aggregates all latest benchmark results, exports CSV, and prints
    summary table.
    """
    latest_csvs = find_latest_case_csvs(bench_dir)
    if not latest_csvs:
        print(f"No benchmark CSVs found in {bench_dir}")
        return None

    stats_list = []
    for case_name, csv_path in latest_csvs.items():
        case_stats = analyze_case_csv(csv_path)
        if case_stats:
            stats_list.append(case_stats)

    if not stats_list:
        print("No valid benchmark rows could be parsed.")
        return None

    # Write aggregate CSV
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv_path = bench_dir / f"bench_stats_{timestamp_str}.csv"

    fieldnames = list(stats_list[0].keys())
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats_list:
            writer.writerow(s)

    # Print summary table
    print("\n" + "=" * 90)
    print(f" FELIX vs PYVALE BENCHMARK SUMMARY ({timestamp_str})")
    print("=" * 90)
    header = (
        f"{'Case Name':<32} | {'Dims':<4} | {'Field':<6} | {'Interp':<6} | "
        f"{'Pyvale (s)':<11} | {'Felix (s)':<10} | {'Speedup':<8}"
    )
    print(header)
    print("-" * 90)

    for s in stats_list:
        c_name = s["case_name"]
        dims = f"{s['spatial_dims']}D"
        f_kind = s["field_kind"][:6]
        interp = "Yes" if str(s["use_temp_interp"]).lower() == "true" else "No"
        py_s = f"{s['pyvale_median_s']:.4f}"
        fx_s = f"{s['felix_median_s']:.4f}"
        spd = f"{s['speedup_median']:.2f}x"
        print(
            f"{c_name:<32} | {dims:<4} | {f_kind:<6} | {interp:<6} | "
            f"{py_s:<11} | {fx_s:<10} | {spd:<8}"
        )

    print("=" * 90)
    print(f"Summary CSV saved to: {summary_csv_path}\n")
    return summary_csv_path


if __name__ == "__main__":
    generate_and_print_summary()
