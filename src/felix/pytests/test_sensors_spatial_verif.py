# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Verification tests for spatial and temporal integration sensors using exact
SymPy symbolic solutions on 2D and 3D finite element meshes.
"""

import numpy as np
import pytest

import felix as fx
import pyvale.verif as verif


def test_sensors_line_scalar_linear_2d() -> None:
    """Tests 1D line sensor measuring average linear scalar field on a
    2D mesh.
    """
    sim_data, data_gen = verif.scalar_linear_2d()

    # Place a horizontal line sensor of length 4.0 centered at (5.0, 3.75, 0.0)
    # The sensor extends from x=3.0 to x=7.0 at constant y=3.75
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
        axis=(1.0, 0.0, 0.0),
        rule=fx.IntegrationGaussLegendre(order=2),
    )

    sensor_array = fx.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=window,
        integration_mode=fx.EIntegrationMode.AVERAGE,
    )

    meas = sensor_array.get_truth()  # shape (1, 1, n_times)

    # Exact symbolic average: (1/L) * integral f(x, y=3.75, t) dx
    for tt, t_val in enumerate(sim_data.time):
        exact_integral = data_gen.integrate_symbolic(
            field_key="temperature",
            bounds_x=(center_x - length / 2.0, center_x + length / 2.0),
            bounds_y=(center_y, center_y),
            bounds_t=(t_val, t_val),
        )
        exact_avg = exact_integral / length
        assert np.isclose(meas[0, 0, tt], exact_avg, rtol=1e-4)


def test_sensors_area_rectangle_scalar_quad_2d() -> None:
    """Tests 2D rectangular area sensor measuring quadratic field on
    a 2D mesh.
    """
    sim_data, data_gen = verif.scalar_quadratic_2d()

    # Rectangular sensor 2.0 x 2.0 centered at (5.0, 3.75, 0.0)
    # x in [4.0, 6.0], y in [2.75, 4.75]
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


def test_sensors_volume_box_scalar_linear_3d() -> None:
    """Tests 3D box volume sensor measuring linear scalar field on a 3D mesh."""
    sim_data, data_gen = verif.scalar_linear_3d()

    # Box 2.0 x 2.0 x 1.0 centered at (5.0, 3.75, 2.5)
    lx, ly, lz = 2.0, 2.0, 1.0
    center = np.array([[5.0, 3.75, 2.5]])
    sens_data = fx.SensorData(positions=center, sample_times=sim_data.time)

    field = fx.FieldScalar(
        sim_data=sim_data,
        comp_key="temperature",
        spatial_dims=fx.EDim.THREED,
    )
    window = fx.SpatialWindowBox(
        length_x=lx,
        length_y=ly,
        length_z=lz,
        rule=fx.IntegrationGaussLegendre(order=2),
    )

    sensor_avg = fx.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=window,
        integration_mode=fx.EIntegrationMode.AVERAGE,
    )

    sensor_acc = fx.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=window,
        integration_mode=fx.EIntegrationMode.ACCUMULATE,
    )

    meas_avg = sensor_avg.get_truth()
    meas_acc = sensor_acc.get_truth()

    vol = lx * ly * lz
    assert np.allclose(meas_acc, meas_avg * vol)

    for tt, t_val in enumerate(sim_data.time):
        exact_integral = data_gen.integrate_symbolic(
            field_key="temperature",
            bounds_x=(5.0 - lx / 2.0, 5.0 + lx / 2.0),
            bounds_y=(3.75 - ly / 2.0, 3.75 + ly / 2.0),
            bounds_z=(2.5 - lz / 2.0, 2.5 + lz / 2.0),
            bounds_t=(t_val, t_val),
        )
        assert np.isclose(meas_acc[0, 0, tt], exact_integral, rtol=1e-4)


def test_sensors_temporal_window_transient_3d() -> None:
    """Tests temporal integration window on dynamic 3D simulation data."""
    sim_data, data_gen = verif.scalar_linear_3d()

    # Time steps from 0.0 to 1.0
    center = np.array([[5.0, 3.75, 2.5]])
    # TemporalWindowRectangular uses [t0 - duration, t0], so sample at t0=0.6
    eval_times = np.array([0.6])

    sens_data = fx.SensorData(positions=center, sample_times=eval_times)
    field = fx.FieldScalar(
        sim_data=sim_data,
        comp_key="temperature",
        spatial_dims=fx.EDim.THREED,
    )

    # Temporal window [0.4, 0.6] -> TemporalWindowRectangular uses [t0 - duration, t0]
    # so t0=0.6, duration=0.2 gives [0.4, 0.6]
    temp_win = fx.TemporalWindowRectangular(
        duration=0.2,
        rule=fx.IntegrationGaussLegendre(order=3),
    )

    sensor = fx.SensorsSpatial(
        sensor_data=sens_data,
        field=field,
        spatial_window=fx.SpatialWindowPoint(),
        temporal_window=temp_win,
        integration_mode=fx.EIntegrationMode.AVERAGE,
    )

    meas = sensor.get_truth()

    # Exact temporal average over [0.4, 0.6]
    exact_integral = data_gen.integrate_symbolic(
        field_key="temperature",
        bounds_x=(5.0, 5.0),
        bounds_y=(3.75, 3.75),
        bounds_z=(2.5, 2.5),
        bounds_t=(0.4, 0.6),
    )
    exact_avg = exact_integral / 0.2
    assert np.isclose(meas[0, 0, 0], exact_avg, rtol=1e-4)
