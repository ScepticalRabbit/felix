# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import numpy as np
import pyvale.verif.analyticsimdatafactory as analytic
import pytest

from felix import EDim, FieldScalar, SensorData, SensorsPoint
from felix import (
    gen_pos_grid_inside,
    gen_pos_grid_boundary,
)
from felix import ErrSysCalibration
from felix import (
    ErrSysRoundOff,
    ErrSysDigitisation,
    ErrSysSaturation,
)
from felix import (
    ErrSysOffset,
    ErrSysOffsetPercent,
)
from felix import GenUniform, GenNormal


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
        cal_range=(0.0, 20.0),
        n_cal_divs=10000,
    )

    sensors = build_analytic_sensors()
    truth = sensors.get_truth()
    sensors.set_error_chain([cal_sim])
    measurements = sensors.sim_measurements()

    raw = (-2.0 + np.sqrt(4.0 + 0.4 * truth)) / 0.2
    expected = assumed_calib(raw)
    assert np.allclose(measurements, expected, atol=1e-3)


def test_sys_errors() -> None:
    sensors = build_analytic_sensors()
    truth = sensors.get_truth()

    sensors.set_error_chain([ErrSysOffset(2.5)])
    assert np.allclose(sensors.sim_measurements(), truth + 2.5)

    sensors.set_error_chain([ErrSysOffsetPercent(5.0)])
    assert np.allclose(sensors.sim_measurements(), truth * 1.05)

    sensors.set_error_chain([ErrSysSaturation(meas_min=0.0, meas_max=8.0)])
    assert np.allclose(sensors.sim_measurements(), np.clip(truth, 0.0, 8.0))


def build_analytic_sensors() -> SensorsPoint:
    sim_data, _ = analytic.scalar_linear_2d()
    field = FieldScalar(sim_data, "temperature", EDim.TWOD)
    sensor_data = SensorData(
        positions=np.array([[2.5, 2.5, 0.0]], dtype=np.float64),
        sample_times=np.array([0.2, 0.7], dtype=np.float64),
    )
    return SensorsPoint(sensor_data, field)
