# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import numpy as np
import pytest

from felix.sensorsim.sensordata import SensorData
from felix.sensorsim.sensortools import (
    gen_pos_grid_inside,
    gen_pos_grid_boundary,
)
from felix.sensorsim.errorsyscalib import ErrSysCalibration
from felix.sensorsim.errorsysdep import (
    ErrSysRoundOff,
    ErrSysDigitisation,
    ErrSysSaturation,
)
from felix.sensorsim.errorsysindep import (
    ErrSysOffset,
    ErrSysOffsetPercent,
)
from felix.sensorsim.generatorsrandom import GenUniform, GenNormal


def test_gen_pos_grid_boundary() -> None:
    grid = gen_pos_grid_boundary(
        num_sensors=(3, 3, 2),
        x_lims=(0.0, 10.0),
        y_lims=(0.0, 20.0),
        z_lims=(0.0, 5.0),
    )
    assert grid.shape == (3 * 3 * 2, 3)
    assert np.min(grid[:, 0]) == pytest.approx(0.0)
    assert np.max(grid[:, 0]) == pytest.approx(10.0)
    assert np.min(grid[:, 1]) == pytest.approx(0.0)
    assert np.max(grid[:, 1]) == pytest.approx(20.0)
    assert np.min(grid[:, 2]) == pytest.approx(0.0)
    assert np.max(grid[:, 2]) == pytest.approx(5.0)


def test_gen_pos_grid_inside() -> None:
    grid = gen_pos_grid_inside(
        num_sensors=(2, 2, 1),
        x_lims=(0.0, 10.0),
        y_lims=(0.0, 20.0),
        z_lims=(0.0, 0.0),
    )
    assert grid.shape == (4, 3)
    # Interior points should not equal boundary min/max
    assert np.min(grid[:, 0]) > 0.0
    assert np.max(grid[:, 0]) < 10.0
    assert np.min(grid[:, 1]) > 0.0
    assert np.max(grid[:, 1]) < 20.0


def test_calibration_inversion() -> None:
    def truth_calib(v: np.ndarray) -> np.ndarray:
        return 2.0 * v + 0.1 * v**2

    def assumed_calib(v: np.ndarray) -> np.ndarray:
        return 2.0 * v

    cal_sim = ErrSysCalibration(
        assumed_calib=assumed_calib,
        truth_calib=truth_calib,
        cal_range=(0.0, 10.0),
        n_cal_divs=10000,
    )

    dummy_sens = SensorData()
    err_basis = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float64)
    errs, _ = cal_sim.sim_errs(err_basis, dummy_sens)

    assert errs.shape == err_basis.shape
    v_exact = (-2.0 + np.sqrt(4.0 + 4.0)) / 0.2
    expected_err = assumed_calib(np.array([v_exact]))[0] - 10.0
    assert errs[0, 0, 0] == pytest.approx(expected_err, abs=1e-3)


def test_sys_errors() -> None:
    sens_data = SensorData()
    basis = np.ones((2, 1, 5)) * 10.0

    # Offset
    e_off = ErrSysOffset(2.5)
    err, _ = e_off.sim_errs(basis, sens_data)
    assert np.allclose(err, 2.5)

    # Offset Percent
    e_pct = ErrSysOffsetPercent(5.0)
    err, _ = e_pct.sim_errs(basis, sens_data)
    assert np.allclose(err, 0.5)

    # Saturation
    e_sat = ErrSysSaturation(meas_min=0.0, meas_max=8.0)
    err, _ = e_sat.sim_errs(basis, sens_data)
    assert np.allclose(basis + err, 8.0)
