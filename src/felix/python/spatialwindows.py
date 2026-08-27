# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Spatial integration windows for finite sensor support domains."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation

from felix.python.enums import EIntegrationMode
from felix.python.integrationrules import (
    IIntegrationRule,
    IntegrationGaussLegendre,
)
from felix.python.spatialkernels import (
    ISpatialKernel,
    SpatialKernelUniform,
)


class ISpatialWindow(ABC):
    """Abstract interface for spatial sensing support windows."""

    @abstractmethod
    def get_spatial_dims(self) -> int:
        """Intrinsic spatial dimension of support window (0, 1, 2, or 3)."""

    @abstractmethod
    def get_measure(self) -> float:
        """Physical geometric measure (length, area, volume)."""

    @abstractmethod
    def get_effective_measure(self) -> float:
        """Integrated effective measure weighted by sensitivity kernel."""

    @abstractmethod
    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates quadrature points and weights in sensor local frame."""

    def to_global_points(
        self,
        sensor_positions: np.ndarray,
        sensor_rotations: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        """Transforms local window integration points to global coordinates."""
        local_pts, _ = self.get_local_points_and_weights()
        n_sensors = sensor_positions.shape[0]
        n_quad = local_pts.shape[0]

        if local_pts.shape[1] == 1:
            local_3d = np.zeros((n_quad, 3))
            local_3d[:, 0] = local_pts[:, 0]
        elif local_pts.shape[1] == 2:
            local_3d = np.zeros((n_quad, 3))
            local_3d[:, :2] = local_pts
        else:
            local_3d = local_pts

        if sensor_rotations is None:
            delta = local_3d[np.newaxis, :, :]
            return sensor_positions[:, np.newaxis, :] + delta

        global_pts = np.empty((n_sensors, n_quad, 3), dtype=float)
        for ii in range(n_sensors):
            rot = (
                sensor_rotations[0]
                if len(sensor_rotations) == 1
                else sensor_rotations[ii]
            )
            rot_pts = rot.apply(local_3d)
            global_pts[ii] = sensor_positions[ii] + rot_pts
        return global_pts


class SpatialWindowPoint(ISpatialWindow):
    """Zero-dimensional point sensor window."""

    __slots__ = ()

    def get_spatial_dims(self) -> int:
        return 0

    def get_measure(self) -> float:
        return 1.0

    def get_effective_measure(self) -> float:
        return 1.0

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((1, 3)), np.ones(1)


class SpatialWindowLine1D(ISpatialWindow):
    """One-dimensional line sensor window."""

    __slots__ = ("_length", "_axis", "_rule", "_kernel")

    def __init__(
        self,
        length: float = 1.0,
        axis: tuple[float, float, float] | np.ndarray = (1.0, 0.0, 0.0),
        rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        self._length = float(length)
        ax = np.asarray(axis, dtype=float)
        norm = np.linalg.norm(ax)
        self._axis = ax / (norm if norm != 0.0 else 1.0)
        self._rule = rule if rule is not None else IntegrationGaussLegendre(2)
        self._kernel = kernel if kernel is not None else SpatialKernelUniform()

    def get_spatial_dims(self) -> int:
        return 1

    def get_measure(self) -> float:
        return self._length

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_canon, weights_canon = self._rule.get_nodes_and_weights(dims=1)
        scale = 0.5 * self._length
        local_xi = nodes_canon[:, 0] * scale
        local_pts = local_xi[:, np.newaxis] * self._axis[np.newaxis, :]

        sens_weights = self._kernel.eval_weights(local_xi[:, np.newaxis])
        int_weights = weights_canon * scale * sens_weights

        if mode == EIntegrationMode.AVERAGE:
            total_w = np.sum(int_weights)
            if total_w != 0.0:
                int_weights = int_weights / total_w
        return local_pts, int_weights


class SpatialWindowRect2D(ISpatialWindow):
    """Two-dimensional rectangular aperture sensor window."""

    __slots__ = ("_length_x", "_length_y", "_rule", "_kernel")

    def __init__(
        self,
        length_x: float = 1.0,
        length_y: float = 1.0,
        rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        self._length_x = float(length_x)
        self._length_y = float(length_y)
        self._rule = rule if rule is not None else IntegrationGaussLegendre(2)
        self._kernel = kernel if kernel is not None else SpatialKernelUniform()

    def get_spatial_dims(self) -> int:
        return 2

    def get_measure(self) -> float:
        return self._length_x * self._length_y

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_canon, weights_canon = self._rule.get_nodes_and_weights(dims=2)
        scale_x = 0.5 * self._length_x
        scale_y = 0.5 * self._length_y
        local_x = nodes_canon[:, 0] * scale_x
        local_y = nodes_canon[:, 1] * scale_y
        local_pts = np.column_stack(
            [local_x, local_y, np.zeros_like(local_x)]
        )

        sens_weights = self._kernel.eval_weights(local_pts[:, :2])
        jacobian = scale_x * scale_y
        int_weights = weights_canon * jacobian * sens_weights

        if mode == EIntegrationMode.AVERAGE:
            total_w = np.sum(int_weights)
            if total_w != 0.0:
                int_weights = int_weights / total_w
        return local_pts, int_weights


class SpatialWindowCircle2D(ISpatialWindow):
    """Two-dimensional circular sensor window (strain gauge pad, disc)."""

    __slots__ = ("_radius", "_rule", "_kernel")

    def __init__(
        self,
        radius: float = 1.0,
        rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        self._radius = float(radius)
        self._rule = rule if rule is not None else IntegrationGaussLegendre(3)
        self._kernel = kernel if kernel is not None else SpatialKernelUniform()

    def get_spatial_dims(self) -> int:
        return 2

    def get_measure(self) -> float:
        return np.pi * self._radius**2

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_canon, weights_canon = self._rule.get_nodes_and_weights(dims=2)
        xi_r = 0.5 * (nodes_canon[:, 0] + 1.0)
        r = self._radius * np.sqrt(xi_r)
        theta = np.pi * (nodes_canon[:, 1] + 1.0)

        local_x = r * np.cos(theta)
        local_y = r * np.sin(theta)
        local_pts = np.column_stack(
            [local_x, local_y, np.zeros_like(local_x)]
        )

        jacobian = 0.5 * np.pi * (self._radius**2)
        sens_weights = self._kernel.eval_weights(local_pts[:, :2])
        int_weights = weights_canon * (0.25 * jacobian) * sens_weights

        if mode == EIntegrationMode.AVERAGE:
            total_w = np.sum(int_weights)
            if total_w != 0.0:
                int_weights = int_weights / total_w
        return local_pts, int_weights


class SpatialWindowSphere3D(ISpatialWindow):
    """Three-dimensional spherical volume sensor window."""

    __slots__ = ("_radius", "_rule", "_kernel")

    def __init__(
        self,
        radius: float = 1.0,
        rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        self._radius = float(radius)
        self._rule = rule if rule is not None else IntegrationGaussLegendre(3)
        self._kernel = kernel if kernel is not None else SpatialKernelUniform()

    def get_spatial_dims(self) -> int:
        return 3

    def get_measure(self) -> float:
        return (4.0 / 3.0) * np.pi * (self._radius**3)

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_canon, weights_canon = self._rule.get_nodes_and_weights(dims=3)
        xi_r = 0.5 * (nodes_canon[:, 0] + 1.0)
        r = self._radius * (xi_r ** (1.0 / 3.0))
        phi = np.pi * (nodes_canon[:, 1] + 1.0)
        cos_theta = nodes_canon[:, 2]
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))

        local_x = r * sin_theta * np.cos(phi)
        local_y = r * sin_theta * np.sin(phi)
        local_z = r * cos_theta
        local_pts = np.column_stack([local_x, local_y, local_z])

        jacobian = (4.0 / 3.0) * np.pi * (self._radius**3)
        sens_weights = self._kernel.eval_weights(local_pts)
        int_weights = weights_canon * (0.125 * jacobian) * sens_weights

        if mode == EIntegrationMode.AVERAGE:
            total_w = np.sum(int_weights)
            if total_w != 0.0:
                int_weights = int_weights / total_w
        return local_pts, int_weights


class SpatialWindowBox3D(ISpatialWindow):
    """Three-dimensional rectangular box volume sensor window."""

    __slots__ = ("_length_x", "_length_y", "_length_z", "_rule", "_kernel")

    def __init__(
        self,
        length_x: float = 1.0,
        length_y: float = 1.0,
        length_z: float = 1.0,
        rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        self._length_x = float(length_x)
        self._length_y = float(length_y)
        self._length_z = float(length_z)
        self._rule = rule if rule is not None else IntegrationGaussLegendre(2)
        self._kernel = kernel if kernel is not None else SpatialKernelUniform()

    def get_spatial_dims(self) -> int:
        return 3

    def get_measure(self) -> float:
        return self._length_x * self._length_y * self._length_z

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_canon, weights_canon = self._rule.get_nodes_and_weights(dims=3)
        scale_x = 0.5 * self._length_x
        scale_y = 0.5 * self._length_y
        scale_z = 0.5 * self._length_z
        local_x = nodes_canon[:, 0] * scale_x
        local_y = nodes_canon[:, 1] * scale_y
        local_z = nodes_canon[:, 2] * scale_z
        local_pts = np.column_stack([local_x, local_y, local_z])

        jacobian = scale_x * scale_y * scale_z
        sens_weights = self._kernel.eval_weights(local_pts)
        int_weights = weights_canon * jacobian * sens_weights

        if mode == EIntegrationMode.AVERAGE:
            total_w = np.sum(int_weights)
            if total_w != 0.0:
                int_weights = int_weights / total_w
        return local_pts, int_weights


SpatialWindowLine = SpatialWindowLine1D
SpatialWindowRectangle = SpatialWindowRect2D
SpatialWindowCircle = SpatialWindowCircle2D
SpatialWindowDisk = SpatialWindowCircle2D
SpatialWindowSphere = SpatialWindowSphere3D
SpatialWindowBox = SpatialWindowBox3D
