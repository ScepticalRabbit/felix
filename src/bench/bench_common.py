# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import csv
from datetime import datetime
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.sensorsim as ps
import felix as fs
from pyvale.dataio.simdata import SimData, EMeshType

from bench.benchparams import (
    SENSORS_NUM,
    SIM_TIMES_NUM,
    SAMPLE_TIMES_NUM,
    MESH_DIVS_2D,
    MESH_DIVS_3D,
    DOMAIN_LENGTH_X,
    DOMAIN_LENGTH_Y,
    DOMAIN_LENGTH_Z,
    CALC_MEAS_CALLS,
    RUNS_PER_CASE,
    WARMUP_RUNS,
    BENCH_OUTPUT_DIR,
)


def create_synthetic_mesh_2d(
    divs: tuple[int, int] = MESH_DIVS_2D,
    length_x: float = DOMAIN_LENGTH_X,
    length_y: float = DOMAIN_LENGTH_Y,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a 2D structured quadrilateral (QUAD4) mesh."""
    num_x = divs[0] + 1
    num_y = divs[1] + 1

    x_vals = np.linspace(0.0, length_x, num_x, dtype=np.float64)
    y_vals = np.linspace(0.0, length_y, num_y, dtype=np.float64)

    xx, yy = np.meshgrid(x_vals, y_vals, indexing="xy")
    coords = np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
            np.zeros(xx.size, dtype=np.float64),
        ]
    )

    quad_elements = []
    for jj in range(divs[1]):
        for ii in range(divs[0]):
            n0 = jj * num_x + ii
            n1 = jj * num_x + (ii + 1)
            n2 = (jj + 1) * num_x + (ii + 1)
            n3 = (jj + 1) * num_x + ii
            quad_elements.append([n0, n1, n2, n3])

    connectivity = np.array(quad_elements, dtype=np.uint64)
    return coords, connectivity


def create_synthetic_mesh_3d(
    divs: tuple[int, int, int] = MESH_DIVS_3D,
    length_x: float = DOMAIN_LENGTH_X,
    length_y: float = DOMAIN_LENGTH_Y,
    length_z: float = DOMAIN_LENGTH_Z,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates a 3D structured hexahedral (HEX8) mesh."""
    num_x = divs[0] + 1
    num_y = divs[1] + 1
    num_z = divs[2] + 1

    x_vals = np.linspace(0.0, length_x, num_x, dtype=np.float64)
    y_vals = np.linspace(0.0, length_y, num_y, dtype=np.float64)
    z_vals = np.linspace(0.0, length_z, num_z, dtype=np.float64)

    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    hex_elements = []
    for kk in range(divs[2]):
        for jj in range(divs[1]):
            for ii in range(divs[0]):
                plane_curr = kk * (num_x * num_y)
                plane_next = (kk + 1) * (num_x * num_y)

                n0 = plane_curr + jj * num_x + ii
                n1 = plane_curr + jj * num_x + (ii + 1)
                n2 = plane_curr + (jj + 1) * num_x + (ii + 1)
                n3 = plane_curr + (jj + 1) * num_x + ii

                n4 = plane_next + jj * num_x + ii
                n5 = plane_next + jj * num_x + (ii + 1)
                n6 = plane_next + (jj + 1) * num_x + (ii + 1)
                n7 = plane_next + (jj + 1) * num_x + ii

                hex_elements.append([n0, n1, n2, n3, n4, n5, n6, n7])

    connectivity = np.array(hex_elements, dtype=np.uint64)
    return coords, connectivity


def create_synthetic_simdata(
    spatial_dims: int,
    field_kind: str,
    num_times: int = SIM_TIMES_NUM,
) -> tuple[SimData, tuple[str, ...]]:
    """Creates a synthetic SimData object with analytical polynomial fields."""
    if spatial_dims == 2:
        coords, connect = create_synthetic_mesh_2d()
        mesh_type = EMeshType.SURF
    else:
        coords, connect = create_synthetic_mesh_3d()
        mesh_type = EMeshType.VOL

    num_nodes = coords.shape[0]
    time_steps = np.linspace(0.0, 1.0, num_times, dtype=np.float64)

    node_vars: dict[str, np.ndarray] = {}
    xx = coords[:, 0:1]
    yy = coords[:, 1:2]
    zz = coords[:, 2:3] if spatial_dims == 3 else np.zeros_like(xx)
    tt = time_steps[np.newaxis, :]

    if field_kind == "scalar":
        comp_keys = ("temperature",)
        node_vars["temperature"] = (
            100.0 + 2.0 * xx + 3.0 * yy + 4.0 * zz
        ) * (1.0 + 0.5 * tt)

    elif field_kind == "vector":
        if spatial_dims == 2:
            comp_keys = ("ux", "uy")
            node_vars["ux"] = (0.01 * xx + 0.02 * yy) * (1.0 + tt)
            node_vars["uy"] = (0.02 * xx + 0.03 * yy) * (1.0 + 0.5 * tt)
        else:
            comp_keys = ("ux", "uy", "uz")
            node_vars["ux"] = (0.01 * xx + 0.02 * yy) * (1.0 + tt)
            node_vars["uy"] = (0.02 * yy + 0.03 * zz) * (1.0 + 0.5 * tt)
            node_vars["uz"] = (0.03 * zz + 0.01 * xx) * (1.0 + 0.25 * tt)

    elif field_kind == "tensor":
        if spatial_dims == 2:
            comp_keys = ("eps_xx", "eps_yy", "eps_xy")
            node_vars["eps_xx"] = (0.001 * xx + 0.002 * yy) * (1.0 + tt)
            node_vars["eps_yy"] = (0.002 * xx + 0.001 * yy) * (1.0 + 0.5 * tt)
            node_vars["eps_xy"] = (0.0005 * (xx + yy)) * (1.0 + 0.2 * tt)
        else:
            comp_keys = (
                "eps_xx",
                "eps_yy",
                "eps_zz",
                "eps_xy",
                "eps_yz",
                "eps_xz",
            )
            node_vars["eps_xx"] = (0.001 * xx) * (1.0 + tt)
            node_vars["eps_yy"] = (0.001 * yy) * (1.0 + 0.5 * tt)
            node_vars["eps_zz"] = (0.001 * zz) * (1.0 + 0.25 * tt)
            node_vars["eps_xy"] = (0.0005 * (xx + yy)) * (1.0 + 0.1 * tt)
            node_vars["eps_yz"] = (0.0005 * (yy + zz)) * (1.0 + 0.1 * tt)
            node_vars["eps_xz"] = (0.0005 * (xx + zz)) * (1.0 + 0.1 * tt)
    else:
        raise ValueError(f"Unknown field kind: {field_kind}")

    connect_dict = (
        {"quad4": connect} if spatial_dims == 2 else {"hex8": connect}
    )

    sim_data = SimData(
        num_spat_dims=spatial_dims,
        mesh_type=mesh_type,
        time=time_steps,
        coords=coords,
        connect=connect_dict,
        node_vars=node_vars,
    )
    return sim_data, comp_keys


def create_sensor_positions(
    spatial_dims: int,
    num_sensors: int = SENSORS_NUM,
    length_x: float = DOMAIN_LENGTH_X,
    length_y: float = DOMAIN_LENGTH_Y,
    length_z: float = DOMAIN_LENGTH_Z,
) -> np.ndarray:
    """Generates deterministic sensor positions distributed in the interior."""
    rng = np.random.default_rng(seed=42)
    margin = 0.05

    px = rng.uniform(
        margin * length_x,
        (1.0 - margin) * length_x,
        size=num_sensors,
    )
    py = rng.uniform(
        margin * length_y,
        (1.0 - margin) * length_y,
        size=num_sensors,
    )

    if spatial_dims == 2:
        pz = np.zeros(num_sensors, dtype=np.float64)
    else:
        pz = rng.uniform(
            margin * length_z,
            (1.0 - margin) * length_z,
            size=num_sensors,
        )

    positions = np.column_stack([px, py, pz])
    return positions


def create_sensor_rotations(
    spatial_dims: int,
    num_sensors: int = SENSORS_NUM,
) -> tuple[Rotation, ...]:
    """Generates deterministic sensor orientation rotations."""
    rng = np.random.default_rng(seed=123)
    if spatial_dims == 2:
        angles_deg = rng.uniform(-45.0, 45.0, size=num_sensors)
        rotations = tuple(
            Rotation.from_euler("z", ang, degrees=True) for ang in angles_deg
        )
    else:
        angles_deg = rng.uniform(-45.0, 45.0, size=(num_sensors, 3))
        rotations = tuple(
            Rotation.from_euler("zyx", ang, degrees=True) for ang in angles_deg
        )
    return rotations


def run_benchmark_case(
    case_name: str,
    spatial_dims: int,
    field_kind: str,
    use_temp_interp: bool,
    num_sensors: int = SENSORS_NUM,
    num_sim_times: int = SIM_TIMES_NUM,
    num_sample_times: int = SAMPLE_TIMES_NUM,
    num_calls: int = CALC_MEAS_CALLS,
    num_runs: int = RUNS_PER_CASE,
    num_warmup: int = WARMUP_RUNS,
) -> Path:
    """Executes a benchmark comparison between Pyvale and Felix."""
    sim_data, comp_keys = create_synthetic_simdata(
        spatial_dims=spatial_dims,
        field_kind=field_kind,
        num_times=num_sim_times,
    )
    positions = create_sensor_positions(
        spatial_dims=spatial_dims,
        num_sensors=num_sensors,
    )

    if use_temp_interp:
        sample_times = np.linspace(
            0.05, 0.95, num_sample_times, dtype=np.float64
        )
    else:
        sample_times = None

    if field_kind in ("vector", "tensor"):
        angles = create_sensor_rotations(
            spatial_dims=spatial_dims,
            num_sensors=num_sensors,
        )
    else:
        angles = None

    edim = ps.EDim.TWOD if spatial_dims == 2 else ps.EDim.THREED

    # Construct Pyvale sensor array
    if field_kind == "scalar":
        field_pyvale = ps.FieldScalar(sim_data, comp_keys[0], edim)
        field_felix = fs.FieldScalar(sim_data, comp_keys[0], edim)
    elif field_kind == "vector":
        field_pyvale = ps.FieldVector(sim_data, comp_keys, edim)
        field_felix = fs.FieldVector(sim_data, comp_keys, edim)
    elif field_kind == "tensor":
        num_norm = 2 if spatial_dims == 2 else 3
        norm_keys = comp_keys[:num_norm]
        dev_keys = comp_keys[num_norm:]
        field_pyvale = ps.FieldTensor(sim_data, norm_keys, dev_keys, edim)
        field_felix = fs.FieldTensor(sim_data, norm_keys, dev_keys, edim)
    else:
        raise ValueError(f"Unknown field kind: {field_kind}")

    sens_data_pyvale = ps.SensorData(
        positions=positions,
        sample_times=sample_times,
        angles=angles,
    )
    sens_data_felix = fs.SensorData(
        positions=positions,
        sample_times=sample_times,
        angles=angles,
    )

    pyvale_sensors = ps.SensorsPoint(sens_data_pyvale, field_pyvale)
    felix_sensors = fs.SensorsPoint(sens_data_felix, field_felix)

    # Warmup runs
    for _ in range(num_warmup):
        pyvale_sensors.calc_truth()
        felix_sensors.calc_truth()

    # Numerical verification check
    truth_pyvale = pyvale_sensors.calc_truth()
    truth_felix = felix_sensors.calc_truth()
    np.testing.assert_allclose(
        truth_felix,
        truth_pyvale,
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"Discrepancy detected in benchmark case {case_name}",
    )

    # Benchmark timing loop
    pyvale_times = []
    felix_times = []

    for _ in range(num_runs):
        # Time Pyvale
        start_py = time.perf_counter()
        for _ in range(num_calls):
            pyvale_sensors.calc_truth()
        time_py = time.perf_counter() - start_py
        pyvale_times.append(time_py)

        # Time Felix
        start_fx = time.perf_counter()
        for _ in range(num_calls):
            felix_sensors.calc_truth()
        time_fx = time.perf_counter() - start_fx
        felix_times.append(time_fx)

    # Output directory setup
    case_dir = BENCH_OUTPUT_DIR / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = case_dir / f"{case_name}_{timestamp_str}.csv"

    fieldnames = [
        "run_idx",
        "case_name",
        "spatial_dims",
        "field_kind",
        "use_temp_interp",
        "sensors_num",
        "sim_times_num",
        "sample_times_num",
        "calc_meas_calls",
        "pyvale_wall_s",
        "felix_wall_s",
        "speedup",
        "pyvale_s_per_call",
        "felix_s_per_call",
    ]

    effective_sample_times = (
        num_sample_times if use_temp_interp else num_sim_times
    )

    with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (t_py, t_fx) in enumerate(zip(pyvale_times, felix_times)):
            speedup = t_py / t_fx if t_fx > 0 else 0.0
            writer.writerow(
                {
                    "run_idx": idx + 1,
                    "case_name": case_name,
                    "spatial_dims": spatial_dims,
                    "field_kind": field_kind,
                    "use_temp_interp": use_temp_interp,
                    "sensors_num": num_sensors,
                    "sim_times_num": num_sim_times,
                    "sample_times_num": effective_sample_times,
                    "calc_meas_calls": num_calls,
                    "pyvale_wall_s": t_py,
                    "felix_wall_s": t_fx,
                    "speedup": speedup,
                    "pyvale_s_per_call": t_py / num_calls,
                    "felix_s_per_call": t_fx / num_calls,
                }
            )

    median_py = np.median(pyvale_times)
    median_fx = np.median(felix_times)
    median_speedup = median_py / median_fx if median_fx > 0 else 0.0

    print(f"[{case_name}] Complete:")
    print(
        f"  Pyvale Median: {median_py:.4f} s "
        f"({median_py/num_calls*1e6:.2f} µs/call)"
    )
    print(
        f"  Felix  Median: {median_fx:.4f} s "
        f"({median_fx/num_calls*1e6:.2f} µs/call)"
    )
    print(f"  Speedup Factor: {median_speedup:.2f}x")
    print(f"  Output CSV:    {csv_file_path}")

    return csv_file_path
