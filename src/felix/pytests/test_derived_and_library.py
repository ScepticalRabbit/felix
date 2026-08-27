# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Tests for derived field transformations, post-processors, and library."""

import numpy as np
import pytest
import pyvale.verif as verif

import felix as fx


def test_field_transform_von_mises_2d() -> None:
    sim_data, _ = verif.tensor_linear_2d()
    field_tensor = fx.FieldTensor(
        sim_data,
        comp_keys=("strain_xx", "strain_yy", "strain_xy"),
        spatial_dims=fx.EDim.TWOD,
    )
    field_vm = fx.FieldTransformed(
        base_field=field_tensor,
        transform=fx.FieldTransformVonMises(),
    )

    pos = np.array([[5.0, 3.75, 0.0]])
    sens_data = fx.SensorData(positions=pos, sample_times=sim_data.time)
    sensor = fx.SensorsPoint(sens_data, field_vm)

    vm_meas = sensor.get_truth()
    assert vm_meas.shape == (1, 1, len(sim_data.time))
    assert np.all(vm_meas >= 0.0)


def test_field_transform_principal_and_hydrostatic_3d() -> None:
    sim_data, _ = verif.tensor_linear_3d()
    comp_keys = (
        "strain_xx",
        "strain_yy",
        "strain_zz",
        "strain_yz",
        "strain_xz",
        "strain_xy",
    )
    field_tensor = fx.FieldTensor(sim_data, comp_keys, fx.EDim.THREED)

    field_p1 = fx.FieldTransformed(
        field_tensor, fx.FieldTransformPrincipal(1)
    )
    field_p3 = fx.FieldTransformed(
        field_tensor, fx.FieldTransformPrincipal(3)
    )
    field_hydro = fx.FieldTransformed(
        field_tensor, fx.FieldTransformHydrostatic()
    )

    pos = np.array([[5.0, 3.75, 2.5]])
    sens_data = fx.SensorData(positions=pos, sample_times=sim_data.time)

    s_p1 = fx.SensorsPoint(sens_data, field_p1).get_truth()
    s_p3 = fx.SensorsPoint(sens_data, field_p3).get_truth()
    s_hydro = fx.SensorsPoint(sens_data, field_hydro).get_truth()

    assert np.all(s_p1 >= s_p3)
    assert s_hydro.shape == (1, 1, len(sim_data.time))


def test_process_stiffness_and_difference() -> None:
    times = np.linspace(0.0, 1.0, 10)
    pos = np.zeros((1, 3))

    f_vals = np.ones((1, 1, 10)) * 100.0
    u_vals = np.ones((1, 1, 10)) * 2.0

    m_f = fx.MeasurementData(
        values=f_vals,
        sample_times=times,
        positions=pos,
        components=("force",),
    )
    m_u = fx.MeasurementData(
        values=u_vals,
        sample_times=times,
        positions=pos,
        components=("displacement",),
    )

    proc_k = fx.ProcessStiffness(force="F", disp="u")
    res_k = proc_k.process({"F": m_f, "u": m_u})
    assert np.allclose(res_k.values, 50.0)

    proc_diff = fx.ProcessRelativeDifference(source_a="A", source_b="B")
    res_diff = proc_diff.process({"A": m_u, "B": m_f})
    assert np.allclose(res_diff.values, 98.0)


def test_sensor_library_presets() -> None:
    sim_scal, _ = verif.scalar_linear_2d()
    sim_vec, _ = verif.vector_linear_2d()
    sim_tens, _ = verif.tensor_linear_2d()

    pos = np.array([[5.0, 3.75, 0.0]])

    # 1. Thermocouple
    tc = fx.SensorLibrary.thermocouple(
        sim_scal, pos, with_meas_errs=True, sample_times=sim_scal.time
    )
    meas_tc = tc.get_measurements()
    assert meas_tc.shape == (1, 1, len(sim_scal.time))

    # 2. Strain gauge
    sg = fx.SensorLibrary.strain_gauge_uniaxial(
        sim_tens, pos, with_meas_errs=True, sample_times=sim_tens.time
    )
    meas_sg = sg.get_measurements()
    assert meas_sg.shape == (1, 3, len(sim_tens.time))

    # 3. Extensometer
    ext = fx.SensorLibrary.extensometer(
        sim_vec,
        pos_a=np.array([[2.0, 2.0, 0.0]]),
        pos_b=np.array([[4.0, 2.0, 0.0]]),
        sample_times=sim_vec.time,
        with_meas_errs=True,
    )
    meas_ext = ext.get_measurements()
    assert meas_ext.shape == (1, 1, len(sim_vec.time))

    # 4. Laser distance
    lidar = fx.SensorLibrary.laser_distance(
        sim_scal,
        origins=np.array([[5.0, 3.75, 50.0]]),
        directions=np.array([[0.0, 0.0, -1.0]]),
        sample_times=sim_scal.time,
        with_meas_errs=True,
    )
    meas_lidar = lidar.get_measurements()
    assert meas_lidar.shape == (1, 1, len(sim_scal.time))
