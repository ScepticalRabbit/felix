# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Verification tests for spatial, ray, and differential sensor arrays."""

import numpy as np
import pytest

import felix as fx
import pyvale.verif as verif


def test_sensors_spatial_line_scalar_2d() -> None:
    """1D line sensor measuring average linear scalar field on 2D mesh."""
    sim_data, data_gen = verif.scalar_linear_2d()

    length = 4.0
    center_x, center_y = 5.0, 3.75
    pos = np.array([[center_x, center_y, 0.0]])

    sens_data = fx.SensorData(positions=pos, sample_times=sim_data.time)
    field = fx.FieldScalar(
        sim_data=sim_data,
        comp_key="temperature",
        spatial_dims=fx.EDim.TWOD,
    )
    window = fx.SpatialWindowLine(
        length=length,
        rule=fx.IntegrationGaussLegendre(order=2),
    )

    sensor_array = fx.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=window,
        integration_mode=fx.EIntegrationMode.AVERAGE,
    )

    meas = sensor_array.get_truth()

    for tt, t_val in enumerate(sim_data.time):
        exact_integral = data_gen.integrate_symbolic(
            field_key="temperature",
            bounds_x=(center_x - length / 2.0, center_x + length / 2.0),
            bounds_y=(center_y, center_y),
            bounds_t=(t_val, t_val),
        )
        exact_avg = exact_integral / length
        assert np.isclose(meas[0, 0, tt], exact_avg, rtol=1e-4)


def test_sensors_spatial_rect_scalar_2d() -> None:
    """2D rectangular sensor measuring quadratic field on 2D mesh."""
    sim_data, data_gen = verif.scalar_quadratic_2d()

    lx, ly = 2.0, 2.0
    center = np.array([[5.0, 3.75, 0.0]])
    sens_data = fx.SensorData(positions=center, sample_times=sim_data.time)

    field = fx.FieldScalar(
        sim_data=sim_data,
        comp_key="temperature",
        spatial_dims=fx.EDim.TWOD,
    )
    window = fx.SpatialWindowRectangle(
        length_x=lx,
        length_y=ly,
        rule=fx.IntegrationGaussLegendre(order=3),
    )

    sensor_array = fx.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=window,
        integration_mode=fx.EIntegrationMode.AVERAGE,
    )

    meas = sensor_array.get_truth()
    area = lx * ly

    for tt, t_val in enumerate(sim_data.time):
        exact_integral = data_gen.integrate_symbolic(
            field_key="temperature",
            bounds_x=(5.0 - lx / 2.0, 5.0 + lx / 2.0),
            bounds_y=(3.75 - ly / 2.0, 3.75 + ly / 2.0),
            bounds_t=(t_val, t_val),
        )
        exact_avg = exact_integral / area
        assert np.isclose(meas[0, 0, tt], exact_avg, rtol=1e-3)


def test_sensors_differential_strain() -> None:
    """Differential sensor array measuring engineering strain."""
    sim_data, _ = verif.vector_linear_2d()

    pos_a = np.array([[2.0, 2.0, 0.0]])
    pos_b = np.array([[4.0, 2.0, 0.0]])

    sens_data_a = fx.SensorData(positions=pos_a, sample_times=sim_data.time)
    sens_data_b = fx.SensorData(positions=pos_b, sample_times=sim_data.time)

    field = fx.FieldVector(
        sim_data=sim_data,
        comp_keys=("disp_x", "disp_y"),
        spatial_dims=fx.EDim.TWOD,
    )

    sens_a = fx.SensorsPoint(sens_data_a, field)
    sens_b = fx.SensorsPoint(sens_data_b, field)

    diff_sensor = fx.SensorsDifferential(
        sensor_a=sens_a,
        sensor_b=sens_b,
        mode=fx.EDifferentialMode.STRAIN,
    )

    strain_meas = diff_sensor.get_truth()
    assert strain_meas.shape == (1, 1, len(sim_data.time))

    truth_a = sens_a.get_truth()
    truth_b = sens_b.get_truth()
    expected_strain = (truth_b[0, 0, :] - truth_a[0, 0, :]) / 2.0
    assert np.allclose(strain_meas[0, 0, :], expected_strain)


def test_sensors_ray_distance() -> None:
    """Ray casting distance measurement."""
    sim_data, _ = verif.scalar_linear_2d()

    ray_origins = np.array([[5.0, 3.75, 50.0]])
    ray_dirs = np.array([[0.0, 0.0, -1.0]])

    ray_sensor = fx.SensorsRay(
        sim_data=sim_data,
        ray_origins=ray_origins,
        ray_directions=ray_dirs,
        max_distance=200.0,
        mode=fx.ERayMode.DISTANCE,
    )

    dist = ray_sensor.get_truth()
    assert np.isclose(dist[0, 0, 0], 50.0, atol=1e-3)
