import numpy as np
import pyvale.sensorsim as py_sens
import pyvale.verif.analyticsimdatafactory as analytic
import pytest
from scipy.spatial.transform import Rotation

import felix


def build_scalar_sensors() -> tuple[
    felix.SensorsPoint,
    object,
    np.ndarray,
    np.ndarray,
]:
    sim_data, data_gen = analytic.scalar_linear_2d()
    positions = np.array(
        [[2.5, 2.5, 0.0], [5.0, 3.75, 0.0]],
        dtype=np.float64,
    )
    times = np.array([0.2, 0.7], dtype=np.float64)
    field = felix.FieldScalar(
        sim_data,
        "temperature",
        felix.EDim.TWOD,
    )
    sensors = felix.SensorsPoint(
        felix.SensorData(positions, times),
        field,
    )
    return sensors, data_gen, positions, times


def test_nominal_sampling_matches_pyvale_analytic_truth() -> None:
    sensors, data_gen, positions, times = build_scalar_sensors()
    expected = data_gen.evaluate_field_truth(
        "temperature",
        positions,
        times,
    )
    assert np.allclose(sensors.sim_measurements()[:, 0, :], expected)


def test_field_position_time_locks_and_dependent_chain() -> None:
    sensors, data_gen, positions, times = build_scalar_sensors()
    pos_offset = np.full_like(positions, 0.2)
    pos_lock = np.zeros_like(positions, dtype=bool)
    pos_lock[:, 0] = True
    pos_lock[:, 2] = True
    field_data = felix.ErrFieldData(
        pos_offset_xyz=pos_offset,
        pos_lock_xyz=pos_lock,
        time_offset=np.full_like(times, 0.1),
    )
    field_error = felix.ErrSysField(sensors.get_field(), field_data)
    sensors.set_error_chain([field_error, field_error])

    expected_positions = positions.copy()
    expected_positions[:, 1] += 0.4
    expected = data_gen.evaluate_field_truth(
        "temperature",
        expected_positions,
        times + 0.2,
    )
    assert np.allclose(sensors.sim_measurements()[:, 0, :], expected)


def test_field_time_drift_is_applied_before_resampling() -> None:
    sensors, data_gen, positions, times = build_scalar_sensors()
    drift = felix.DriftLinear(rate=0.1, time_start=0.0, offset=0.05)
    field_error = felix.ErrSysField(
        sensors.get_field(),
        felix.ErrFieldData(time_drift=drift),
    )
    sensors.set_error_chain([field_error])

    expected_times = times + 0.1 * times + 0.05
    expected = data_gen.evaluate_field_truth(
        "temperature",
        positions,
        expected_times,
    )
    assert np.allclose(sensors.sim_measurements()[:, 0, :], expected)


def test_field_angle_perturbation_matches_pyvale_vector_transform() -> None:
    sim_data, _ = analytic.vector_linear_2d()
    positions = np.array([[2.5, 2.5, 0.0]], dtype=np.float64)
    times = np.array([0.2, 0.7], dtype=np.float64)
    angle_offset = np.array([[30.0, 0.0, 0.0]], dtype=np.float64)
    field = felix.FieldVector(
        sim_data,
        ("disp_x", "disp_y"),
        felix.EDim.TWOD,
    )
    sensors = felix.SensorsPoint(felix.SensorData(positions, times), field)
    sensors.set_error_chain(
        [
            felix.ErrSysField(
                field,
                felix.ErrFieldData(ang_offset_zyx=angle_offset),
            )
        ]
    )

    py_field = py_sens.FieldVector(
        sim_data,
        ("disp_x", "disp_y"),
        py_sens.EDim.TWOD,
    )
    rotation = (Rotation.from_euler("zyx", angle_offset[0], degrees=True),)
    expected = py_field.sample_field(positions, times, rotation)
    assert np.allclose(sensors.sim_measurements(), expected)


def test_spatial_average_matches_pyvale_analytic_example() -> None:
    sim_data, _ = analytic.scalar_quadratic_2d()
    positions = np.array([[5.0, 3.75, 0.0]], dtype=np.float64)
    times = np.array([0.5], dtype=np.float64)
    dimensions = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    field = felix.FieldScalar(sim_data, "temperature", felix.EDim.TWOD)
    sensors = felix.SensorsPoint(felix.SensorData(positions, times), field)
    sensors.set_error_chain(
        [
            felix.ErrSysField(
                field,
                felix.ErrFieldData(
                    spatial_averager=py_sens.EIntSpatialType.QUAD9PT,
                    spatial_dims=dimensions,
                ),
            )
        ]
    )

    py_field = py_sens.FieldScalar(
        sim_data,
        "temperature",
        py_sens.EDim.TWOD,
    )
    py_data = py_sens.SensorData(
        positions=positions,
        sample_times=times,
        spatial_averager=py_sens.EIntSpatialType.QUAD9PT,
        spatial_dims=dimensions,
    )
    expected = py_sens.SensorsPoint(py_data, py_field).sim_measurements()
    assert np.allclose(sensors.sim_measurements(), expected, atol=1e-10)


def test_zig_random_distribution_has_expected_coarse_statistics() -> None:
    sensors, _, _, _ = build_scalar_sensors()
    sensors.set_error_chain(
        [felix.ErrRandGen(felix.GenNormal(mean=2.0, std=3.0, seed=19))]
    )
    errors = sensors.sim_experiments(5000, seed=100)[4]
    assert np.mean(errors) == pytest.approx(2.0, abs=0.12)
    assert np.std(errors) == pytest.approx(3.0, abs=0.12)


def test_experiment_statistics_execute_through_zig() -> None:
    values = np.array(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        dtype=np.float64,
    )
    stats = felix.calc_exp_sim_stats(values)
    assert np.allclose(stats.mean, np.mean(values, axis=0))
    assert np.allclose(stats.std, np.std(values, axis=0))
    assert np.allclose(stats.var, np.var(values, axis=0))
    assert np.allclose(stats.median, np.median(values, axis=0))
    expected_mad = np.median(
        np.abs(values - np.median(values, axis=0)),
        axis=0,
    )
    assert np.allclose(stats.mad, expected_mad)
