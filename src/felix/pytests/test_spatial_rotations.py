# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from felix import (
    orient_from_direction,
    orient_from_normal_and_tangent,
)


def test_orient_from_direction() -> None:
    targets = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-2.0, 3.5, 1.2],
    ]
    e1 = np.array([1.0, 0.0, 0.0])

    for t in targets:
        rot = orient_from_direction(t)
        transformed = rot.apply(e1)
        expected = np.array(t) / np.linalg.norm(t)
        np.testing.assert_allclose(transformed, expected, atol=1e-6)


def test_orient_from_normal_and_tangent() -> None:
    normal = (0.0, 0.0, 1.0)
    tangent = (1.0, 1.0, 0.0)

    rot = orient_from_normal_and_tangent(normal, tangent)
    matrix = rot.as_matrix()

    e1 = matrix[:, 0]
    e2 = matrix[:, 1]
    e3 = matrix[:, 2]

    assert np.isclose(np.dot(e1, e2), 0.0, atol=1e-7)
    assert np.isclose(np.dot(e2, e3), 0.0, atol=1e-7)
    assert np.isclose(np.dot(e1, e3), 0.0, atol=1e-7)

    assert np.allclose(e3, [0.0, 0.0, 1.0])
    assert np.allclose(e1, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))


def test_tensor_rotation_transformation() -> None:
    sigma_global = np.array([
        [100.0, 0.0, 0.0],
        [0.0, 50.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    rot = Rotation.from_euler("z", 45.0, degrees=True)
    R = rot.as_matrix()
    sigma_local = R.T @ sigma_global @ R

    assert np.isclose(sigma_local[0, 0], 75.0)
    assert np.isclose(sigma_local[1, 1], 75.0)
    assert np.isclose(sigma_local[0, 1], -25.0)
    assert np.isclose(sigma_local[1, 0], -25.0)


def test_vector_2d_transform_with_felix_field() -> None:
    from pyvale.dataio.simdata import SimData
    import felix as fx

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    connect = {"quad4": np.array([[0, 1, 2, 3]], dtype=np.int64)}
    node_vars = {
        "disp_x": np.array([[10.0], [10.0], [10.0], [10.0]]),
        "disp_y": np.array([[0.0], [0.0], [0.0], [0.0]]),
    }
    sim_data = SimData(
        num_spat_dims=2,
        mesh_type=None,
        time=np.array([0.0]),
        coords=coords,
        connect=connect,
        node_vars=node_vars,
    )
    field = fx.FieldVector(sim_data, ("disp_x", "disp_y"), fx.EDim.TWOD)
    rot = (Rotation.from_euler("z", 90.0, degrees=True),)
    sens_data = fx.SensorData(
        positions=np.array([[0.5, 0.5, 0.0]]),
        angles=rot,
    )
    sensors = fx.SensorsPoint(sens_data, field)
    truth = sensors.calc_truth()
    # Rotated by 90 deg: local x = global y = 0, local y = -global x = -10
    assert np.isclose(truth[0, 0, 0], 0.0, atol=1e-6)
    assert np.isclose(truth[0, 1, 0], -10.0, atol=1e-6)


def test_tensor_2d_transform_with_felix_field() -> None:
    from pyvale.dataio.simdata import SimData
    import felix as fx

    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    connect = {"quad4": np.array([[0, 1, 2, 3]], dtype=np.int64)}
    node_vars = {
        "eps_xx": np.array([[100.0], [100.0], [100.0], [100.0]]),
        "eps_yy": np.array([[50.0], [50.0], [50.0], [50.0]]),
        "eps_xy": np.array([[0.0], [0.0], [0.0], [0.0]]),
    }
    sim_data = SimData(
        num_spat_dims=2,
        mesh_type=None,
        time=np.array([0.0]),
        coords=coords,
        connect=connect,
        node_vars=node_vars,
    )
    field = fx.FieldTensor(
        sim_data, ("eps_xx", "eps_yy"), ("eps_xy",), fx.EDim.TWOD
    )
    rot = (Rotation.from_euler("z", 45.0, degrees=True),)
    sens_data = fx.SensorData(
        positions=np.array([[0.5, 0.5, 0.0]]),
        angles=rot,
    )
    sensors = fx.SensorsPoint(sens_data, field)
    truth = sensors.calc_truth()
    assert np.isclose(truth[0, 0, 0], 75.0, atol=1e-6)
    assert np.isclose(truth[0, 1, 0], 75.0, atol=1e-6)
    assert np.isclose(truth[0, 2, 0], -25.0, atol=1e-6)
