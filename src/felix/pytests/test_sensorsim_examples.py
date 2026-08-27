# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import runpy
from pathlib import Path

import pytest
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
pv.OFF_SCREEN = True

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASIC_EXS_DIR = PROJECT_ROOT / "src" / "felix" / "pyexs_basic"
EXT_EXS_DIR = PROJECT_ROOT / "src" / "felix" / "pyexs_ext"

BASIC_EXAMPLES = [
    "ex0_quickstart.py",
    "ex1_scalar_sensors.py",
    "ex2_vector_tensor_sensors.py",
    "ex3_experiment_simulator.py",
]

EXTENDED_EXAMPLES = [
    "ex1_byosimdata.py",
    "ex3a_scal2d.py",
    "ex3b_scal3d.py",
    "ex3c_vec2d.py",
    "ex3d_vec3d.py",
    "ex3e_tens2d.py",
    "ex3f_tens3d.py",
    "ex4a_basicerrs_scal2d.py",
    "ex4b_fielderrs_scal3d.py",
    "ex4c_angleerrs_vec2d.py",
    "ex4d_fieldlockerrs_vec3d.py",
    "ex4e_chainfielderrs_vec2d.py",
    "ex4f_caliberrs_scal2d.py",
    "ex5a_expsim_thermmech2d.py",
    "ex5b_expsim_thermmech3d.py",
    "ex6a_errgraph_parallel_branches.py",
    "ex6b_errgraph_diamond_dependency.py",
    "ex6c_errgraph_chain_conversion_and_inspection.py",
    "ex7a_line_sensors_fbg_fiber.py",
    "ex7b_area_sensors_foil_strain_gauge.py",
    "ex7d_spatial_kernels_gaussian_psf.py",
    "ex7e_temporal_windowing_shutter_lag.py",
    "ex8a_derived_fields_stress_invariants.py",
    "ex8c_flux_sensors_heat_and_flow.py",
    "ex8d_differential_sensors_extensometer.py",
    "ex8e_ray_sensors_lidar_and_pyrometer.py",
    "ex8f_sensor_library_typical_transducers.py",
]


@pytest.mark.parametrize("example_name", BASIC_EXAMPLES)
def test_basic_sensorsim_example(
    example_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    example_path = BASIC_EXS_DIR / example_name
    assert example_path.exists(), f"Example {example_name} not found"
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(example_path), run_name="__main__")


@pytest.mark.parametrize("example_name", EXTENDED_EXAMPLES)
def test_extended_sensorsim_example(
    example_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    example_path = EXT_EXS_DIR / example_name
    assert example_path.exists(), f"Example {example_name} not found"
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(example_path), run_name="__main__")
