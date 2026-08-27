# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Tests for weighting kernels and numerical quadrature in Felix."""

import numpy as np
import pytest

import felix as fx


def test_spatial_kernel_uniform() -> None:
    coords = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    k = fx.SpatialKernelUniform()
    w = k.eval_weights(coords)
    assert np.allclose(w, np.ones(3))


def test_spatial_kernel_gaussian() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    k = fx.SpatialKernelGaussian(sigma=1.0)
    w = k.eval_weights(coords)
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], np.exp(-0.5))


def test_spatial_kernel_triangular() -> None:
    coords = np.array([[0.0, 0.0], [0.5, 0.0], [1.5, 0.0]])
    k = fx.SpatialKernelTriangular(radii=1.0)
    w = k.eval_weights(coords)
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], 0.5)
    assert np.isclose(w[2], 0.0)


def test_spatial_kernel_cosine() -> None:
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    k = fx.SpatialKernelCosine(radius=1.0)
    w = k.eval_weights(coords)
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], 0.0)


def test_spatial_kernel_epanechnikov() -> None:
    coords = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    k = fx.SpatialKernelEpanechnikov(radius=1.0)
    w = k.eval_weights(coords)
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], 0.75)
    assert np.isclose(w[2], 0.0)


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_gauss_legendre_quadrature(dims: int) -> None:
    rule = fx.IntegrationGaussLegendre(order=3)
    nodes, weights = rule.get_nodes_and_weights(dims=dims)

    expected_pts = 3**dims
    assert nodes.shape == (expected_pts, dims)
    assert weights.shape == (expected_pts,)

    # Total weight on [-1, 1]^d must equal 2^d
    expected_vol = 2.0**dims
    assert np.isclose(np.sum(weights), expected_vol)


@pytest.mark.parametrize("dims", [1, 2])
def test_trapezoidal_quadrature(dims: int) -> None:
    rule = fx.IntegrationTrapezoidal(divisions=4)
    nodes, weights = rule.get_nodes_and_weights(dims=dims)

    expected_pts = 5**dims
    assert nodes.shape == (expected_pts, dims)
    assert weights.shape == (expected_pts,)

    expected_vol = 2.0**dims
    assert np.isclose(np.sum(weights), expected_vol)


@pytest.mark.parametrize("dims", [1, 2])
def test_simpson_quadrature(dims: int) -> None:
    rule = fx.IntegrationSimpson(divisions=4)
    nodes, weights = rule.get_nodes_and_weights(dims=dims)

    expected_pts = 5**dims
    assert nodes.shape == (expected_pts, dims)
    assert weights.shape == (expected_pts,)

    expected_vol = 2.0**dims
    assert np.isclose(np.sum(weights), expected_vol)
