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
from typing import Any

import felix as fx
import pytest
import matplotlib
matplotlib.use("Agg")

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
    "ex7a_line_sensors_fbg_fiber.py",
    "ex7b_area_sensors_foil_strain_gauge.py",
    "ex7d_spatial_kernels_gaussian_psf.py",
    "ex7e_temporal_windowing_shutter_lag.py",
    "ex8a_derived_fields_stress_invariants.py",
    "ex8d_differential_sensors_extensometer.py",
    "ex8e_ray_sensors_lidar_and_pyrometer.py",
    "ex8f_sensor_library_typical_transducers.py",
]


class _PlotterDisabled:
    """Minimal plotter used while executing examples under pytest."""

    off_screen = True
    camera_position: Any = None

    def screenshot(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def show(self) -> None:
        return None


class _FigureDisabled:
    def savefig(self, *args: Any, **kwargs: Any) -> None:
        return None


@pytest.fixture(autouse=True)
def disable_pyvista_plotting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise example calculations without creating native VTK windows."""
    def create_disabled_plotter(*args: Any, **kwargs: Any) -> _PlotterDisabled:
        return _PlotterDisabled()

    def create_disabled_figure(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        return _FigureDisabled(), None

    monkeypatch.setattr(
        fx,
        "plot_point_sensors_on_sim",
        create_disabled_plotter,
    )
    monkeypatch.setattr(
        fx,
        "plot_sensors_on_sim",
        create_disabled_plotter,
    )
    monkeypatch.setattr(
        fx,
        "plot_time_traces",
        create_disabled_figure,
    )
    monkeypatch.setattr(
        fx,
        "plot_exp_traces",
        create_disabled_figure,
    )


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
