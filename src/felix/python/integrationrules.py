# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Numerical integration quadrature rules in Felix."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

import felix.cython.felix as fc


class IIntegrationRule(ABC):
    """Abstract base interface for N-dimensional numerical integration rules."""

    @abstractmethod
    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates integration nodes and weights on canonical domain [-1, 1]^d."""


class IntegrationGaussLegendre(IIntegrationRule):
    """Gauss-Legendre quadrature rule on [-1, 1]^d."""

    __slots__ = ("_order",)

    def __init__(self, order: int = 2) -> None:
        self._order = int(order)

    def get_order(self) -> int:
        return self._order

    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return fc.generate_quadrature_rule(0, self._order, dims)


class IntegrationMidpoint(IIntegrationRule):
    """Uniform piecewise midpoint integration rule on [-1, 1]^d."""

    __slots__ = ("_divisions",)

    def __init__(self, divisions: int = 4) -> None:
        self._divisions = int(divisions)

    def get_divisions(self) -> int:
        return self._divisions

    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return fc.generate_quadrature_rule(1, self._divisions, dims)


class IntegrationTrapezoidal(IIntegrationRule):
    """Composite trapezoidal integration rule on [-1, 1]^d."""

    __slots__ = ("_divisions",)

    def __init__(self, divisions: int = 4) -> None:
        self._divisions = int(divisions)

    def get_divisions(self) -> int:
        return self._divisions

    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return fc.generate_quadrature_rule(2, self._divisions, dims)


class IntegrationSimpson(IIntegrationRule):
    """Composite Simpson's rule on [-1, 1]^d."""

    __slots__ = ("_divisions",)

    def __init__(self, divisions: int = 4) -> None:
        self._divisions = int(divisions)

    def get_divisions(self) -> int:
        return self._divisions

    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return fc.generate_quadrature_rule(3, self._divisions, dims)


class IntegrationMonteCarlo(IIntegrationRule):
    """Uniform quasi-Monte Carlo integration on [-1, 1]^d."""

    __slots__ = ("_num_samples", "_seed")

    def __init__(
        self, num_samples: int = 100, seed: int | None = None
    ) -> None:
        self._num_samples = int(num_samples)
        self._seed = seed

    def get_num_samples(self) -> int:
        return self._num_samples

    def get_nodes_and_weights(
        self, dims: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self._seed)
        nodes = rng.uniform(-1.0, 1.0, size=(self._num_samples, dims))
        domain_volume = 2.0**dims
        weights = np.full(self._num_samples, domain_volume / self._num_samples)
        return nodes, weights
