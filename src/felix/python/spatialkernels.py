# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Spatial sensitivity weighting kernels in Felix."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np

import felix.cython.felix as fc


class ISpatialKernel(ABC):
    """Abstract interface for continuous spatial sensitivity kernels."""

    @abstractmethod
    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        """Evaluates sensitivity weights at local coordinates."""


class SpatialKernelUniform(ISpatialKernel):
    """Uniform spatial sensitivity kernel w(x) = 1.0."""

    __slots__ = ()

    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        return fc.eval_kernel_weights(0, local_coords)


class SpatialKernelGaussian(ISpatialKernel):
    """Continuous Gaussian sensitivity kernel."""

    __slots__ = ("_sigma",)

    def __init__(self, sigma: float | tuple[float, ...]) -> None:
        self._sigma = sigma

    def get_sigma(self) -> float | tuple[float, ...]:
        return self._sigma

    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        if isinstance(self._sigma, (int, float)):
            sig0, sig1, sig2 = float(self._sigma), 0.0, 0.0
        else:
            sig_list = list(self._sigma)
            sig0 = float(sig_list[0]) if len(sig_list) > 0 else 1.0
            sig1 = float(sig_list[1]) if len(sig_list) > 1 else sig0
            sig2 = float(sig_list[2]) if len(sig_list) > 2 else sig0
        return fc.eval_kernel_weights(1, local_coords, sig0, sig1, sig2)


class SpatialKernelTriangular(ISpatialKernel):
    """Triangular (conical) sensitivity kernel decaying to zero."""

    __slots__ = ("_radii",)

    def __init__(self, radii: float | tuple[float, ...]) -> None:
        self._radii = radii

    def get_radii(self) -> float | tuple[float, ...]:
        return self._radii

    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        if isinstance(self._radii, (int, float)):
            rad0, rad1, rad2 = float(self._radii), 0.0, 0.0
        else:
            rad_list = list(self._radii)
            rad0 = float(rad_list[0]) if len(rad_list) > 0 else 1.0
            rad1 = float(rad_list[1]) if len(rad_list) > 1 else rad0
            rad2 = float(rad_list[2]) if len(rad_list) > 2 else rad0
        return fc.eval_kernel_weights(2, local_coords, rad0, rad1, rad2)


class SpatialKernelCosine(ISpatialKernel):
    """Cosine sensitivity kernel."""

    __slots__ = ("_radius",)

    def __init__(self, radius: float = 1.0) -> None:
        self._radius = float(radius)

    def get_radius(self) -> float:
        return self._radius

    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        return fc.eval_kernel_weights(3, local_coords, self._radius)


class SpatialKernelEpanechnikov(ISpatialKernel):
    """Parabolic Epanechnikov sensitivity kernel."""

    __slots__ = ("_radius",)

    def __init__(self, radius: float = 1.0) -> None:
        self._radius = float(radius)

    def get_radius(self) -> float:
        return self._radius

    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        return fc.eval_kernel_weights(4, local_coords, self._radius)


class SpatialKernelCustom(ISpatialKernel):
    """Custom callable spatial sensitivity kernel."""

    __slots__ = ("_func",)

    def __init__(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        self._func = func

    def eval_weights(self, local_coords: np.ndarray) -> np.ndarray:
        return np.asarray(self._func(local_coords), dtype=float)
