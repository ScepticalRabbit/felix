# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Temporal response windows and filtering kernels in Felix."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from felix.python.enums import EIntegrationMode
from felix.python.integrationrules import (
    IIntegrationRule,
    IntegrationGaussLegendre,
)


class ITemporalKernel(ABC):
    """Abstract interface for continuous temporal weighting kernels."""

    @abstractmethod
    def eval_weights(self, tau: np.ndarray, duration: float) -> np.ndarray:
        """Evaluates continuous temporal sensitivity weights."""


class TemporalKernelUniform(ITemporalKernel):
    """Uniform temporal weighting kernel."""

    __slots__ = ()

    def eval_weights(self, tau: np.ndarray, duration: float) -> np.ndarray:
        return np.ones_like(tau, dtype=float)


class TemporalKernelExponentialDecay(ITemporalKernel):
    """First-order exponential response lag kernel."""

    __slots__ = ("_time_constant",)

    def __init__(self, time_constant: float) -> None:
        self._time_constant = float(time_constant)

    def get_time_constant(self) -> float:
        return self._time_constant

    def eval_weights(self, tau: np.ndarray, duration: float) -> np.ndarray:
        return np.exp(tau / self._time_constant)


class TemporalKernelGaussian(ITemporalKernel):
    """Gaussian temporal sensitivity kernel."""

    __slots__ = ("_sigma",)

    def __init__(self, sigma: float) -> None:
        self._sigma = float(sigma)

    def get_sigma(self) -> float:
        return self._sigma

    def eval_weights(self, tau: np.ndarray, duration: float) -> np.ndarray:
        return np.exp(-0.5 * (tau / self._sigma) ** 2)


class ITemporalWindow(ABC):
    """Abstract interface for temporal integration and response windows."""

    @abstractmethod
    def get_duration(self) -> float:
        """Total temporal integration window duration."""

    @abstractmethod
    def get_effective_duration(self) -> float:
        """Integrated effective window duration."""

    @abstractmethod
    def get_sample_offsets_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates time offsets tau and weights relative to measurement time."""


class TemporalWindowInstant(ITemporalWindow):
    """Instantaneous zero-duration measurement window."""

    __slots__ = ()

    def get_duration(self) -> float:
        return 0.0

    def get_effective_duration(self) -> float:
        return 1.0

    def get_sample_offsets_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(1), np.ones(1)


class TemporalWindowRectangular(ITemporalWindow):
    """Finite rectangular exposure duration window: [t0 - T, t0]."""

    __slots__ = ("_duration", "_rule", "_kernel")

    def __init__(
        self,
        duration: float = 1.0,
        rule: IIntegrationRule | None = None,
        kernel: ITemporalKernel | None = None,
    ) -> None:
        self._duration = float(duration)
        self._rule = rule if rule is not None else IntegrationGaussLegendre(3)
        self._kernel = (
            kernel if kernel is not None else TemporalKernelUniform()
        )

    def get_duration(self) -> float:
        return self._duration

    def get_effective_duration(self) -> float:
        _, weights = self.get_sample_offsets_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_sample_offsets_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_canon, weights_canon = self._rule.get_nodes_and_weights(dims=1)
        scale = 0.5 * self._duration
        tau = (nodes_canon[:, 0] - 1.0) * scale

        sens_weights = self._kernel.eval_weights(tau, self._duration)
        int_weights = weights_canon * scale * sens_weights

        if mode == EIntegrationMode.AVERAGE:
            total_w = np.sum(int_weights)
            if total_w != 0.0:
                int_weights = int_weights / total_w
        return tau, int_weights


class TemporalWindowExponential(TemporalWindowRectangular):
    """Exponential decay temporal response lag window."""

    def __init__(
        self,
        time_constant: float = 1.0,
        num_time_constants: float = 5.0,
        rule: IIntegrationRule | None = None,
    ) -> None:
        duration = time_constant * num_time_constants
        kernel = TemporalKernelExponentialDecay(time_constant)
        super().__init__(duration=duration, rule=rule, kernel=kernel)


class TemporalWindowGaussian(TemporalWindowRectangular):
    """Gaussian temporal response window."""

    def __init__(
        self,
        sigma: float = 1.0,
        num_sigmas: float = 3.0,
        rule: IIntegrationRule | None = None,
    ) -> None:
        duration = 2.0 * sigma * num_sigmas
        kernel = TemporalKernelGaussian(sigma)
        super().__init__(duration=duration, rule=rule, kernel=kernel)
