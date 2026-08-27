# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
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
