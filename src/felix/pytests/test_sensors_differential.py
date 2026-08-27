# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Unit and verification tests for differential sensors in Felix.
"""

import numpy as np
import pytest

import felix as fx
import pyvale.verif as verif


def test_differential_temperature_difference() -> None:
    sim_data, _ = verif.scalar_linear_2d()

    # Anchor A at (3.0, 3.0, 0.0), Anchor B at (7.0, 3.0, 0.0)
    field_a = fx.FieldScalar(sim_data, "temperature", fx.EDim.TWOD)
    field_b = fx.FieldScalar(sim_data, "temperature", fx.EDim.TWOD)

    sens_data_a = fx.SensorData(positions=np.array([[3.0, 3.0, 0.0]]))
    sens_data_b = fx.SensorData(positions=np.array([[7.0, 3.0, 0.0]]))

    sens_a = fx.SensorsSpatial(sens_data_a, field_a)
    sens_b = fx.SensorsSpatial(sens_data_b, field_b)

    diff_sensor = fx.SensorsDifferential(
        sensor_a=sens_a,
        sensor_b=sens_b,
        mode=fx.EDifferentialMode.DIFFERENCE,
    )

    truth = diff_sensor.get_truth()
    assert truth.shape == (1, 1, sim_data.time.shape[0])
    truth_a = sens_a.get_truth()
    truth_b = sens_b.get_truth()
    assert np.allclose(truth, truth_b - truth_a)


def test_extensometer_strain_mode() -> None:
    sim_data, _ = verif.scalar_linear_2d()

    n_times = sim_data.time.shape[0]
    ux = 0.002 * sim_data.coords[:, 0:1]
    sim_data.node_vars["disp_x"] = np.tile(ux, (1, n_times))
    sim_data.node_vars["disp_y"] = np.zeros(
        (sim_data.coords.shape[0], n_times), dtype=np.float64
    )

    field_a = fx.FieldVector(sim_data, ("disp_x", "disp_y"), fx.EDim.TWOD)
    field_b = fx.FieldVector(sim_data, ("disp_x", "disp_y"), fx.EDim.TWOD)

    # Gauge length = 4.0 mm from x=3.0 to x=7.0
    sens_data_a = fx.SensorData(positions=np.array([[3.0, 3.0, 0.0]]))
    sens_data_b = fx.SensorData(positions=np.array([[7.0, 3.0, 0.0]]))

    s_win = fx.SpatialWindowLine(length=2.0, axis=(0.0, 1.0, 0.0))

    sens_a = fx.SensorsSpatial(sens_data_a, field_a, spatial_window=s_win)
    sens_b = fx.SensorsSpatial(sens_data_b, field_b, spatial_window=s_win)

    ext = fx.SensorsDifferential(
        sensor_a=sens_a,
        sensor_b=sens_b,
        mode=fx.EDifferentialMode.STRAIN,
    )

    truth = ext.get_truth()
    # eps = (u_xB - u_xA) / L0 = (0.002*7 - 0.002*3) / 4.0 = 0.0020
    assert np.isclose(truth[0, 0, 0], 0.0020, atol=1e-5)


def test_differential_errors_and_custom_func() -> None:
    sim_data, _ = verif.scalar_linear_2d()

    field = fx.FieldScalar(sim_data, "temperature", fx.EDim.TWOD)
    sens_data_a = fx.SensorData(positions=np.array([[3.0, 3.0, 0.0]]))
    sens_data_b = fx.SensorData(positions=np.array([[7.0, 3.0, 0.0]]))

    sens_a = fx.SensorsSpatial(sens_data_a, field)
    sens_b = fx.SensorsSpatial(sens_data_b, field)

    custom_fn = lambda a, b: (b - a) / 2.0
    diff_sensor = fx.SensorsDifferential(
        sensor_a=sens_a,
        sensor_b=sens_b,
        mode=fx.EDifferentialMode.CUSTOM,
        custom_func=custom_fn,
    )

    truth = diff_sensor.get_truth()
    truth_a = sens_a.get_truth()
    truth_b = sens_b.get_truth()
    expected = (truth_b - truth_a) / 2.0
    assert np.allclose(truth, expected)

    err_chain = (
        fx.ErrSysOffset(offset=0.5),
        fx.ErrRandGen(fx.GenNormal(std=0.1, mean=0.0)),
    )
    diff_sensor.set_error_chain(err_chain)

    meas = diff_sensor.sim_measurements()
    err_tot = diff_sensor.get_errors_total()
    assert err_tot is not None
    assert np.allclose(meas, truth + err_tot)
