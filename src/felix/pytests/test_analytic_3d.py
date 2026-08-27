# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Unit tests for 3D analytic mesh and SimData generation in felix.verif.
"""

import numpy as np
import pytest
import sympy

import felix as fx
import pyvale.verif as verif


def test_box_mesh_3d_geometry() -> None:
    """Test 3D hexahedral box mesh generation."""
    leng_x, leng_y, leng_z = 10.0, 8.0, 6.0
    n_elem_x, n_elem_y, n_elem_z = 5, 4, 3

    coords, connect = verif.box_mesh_3d(
        leng_x=leng_x,
        leng_y=leng_y,
        leng_z=leng_z,
        n_elem_x=n_elem_x,
        n_elem_y=n_elem_y,
        n_elem_z=n_elem_z,
    )

    n_nodes_expected = (n_elem_x + 1) * (n_elem_y + 1) * (n_elem_z + 1)
    n_elems_expected = n_elem_x * n_elem_y * n_elem_z

    assert coords.shape == (n_nodes_expected, 3)
    assert connect.shape == (n_elems_expected, 8)

    # Check bounds
    assert np.isclose(np.min(coords[:, 0]), 0.0)
    assert np.max(coords[:, 0]) == pytest.approx(leng_x)
    assert np.min(coords[:, 1]) == pytest.approx(0.0)
    assert np.max(coords[:, 1]) == pytest.approx(leng_y)
    assert np.min(coords[:, 2]) == pytest.approx(0.0)
    assert np.max(coords[:, 2]) == pytest.approx(leng_z)

    # Connectivity node index validity
    assert np.min(connect) == 0
    assert np.max(connect) == n_nodes_expected - 1


def test_fill_dims_3d() -> None:
    """Test fill_dims_3d array broadcasting."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([3.0, 4.0, 5.0])
    z = np.array([6.0, 7.0, 8.0])
    time = np.array([0.0, 0.5, 1.0, 1.5])

    fx_c, fy_c, fz_c, ft_c = verif.fill_dims_3d(x, y, z, time)

    assert fx_c.shape == (3, 4)
    assert fy_c.shape == (3, 4)
    assert fz_c.shape == (3, 4)
    assert ft_c.shape == (3, 4)
    assert np.allclose(fx_c[:, 0], x)
    assert np.allclose(ft_c[0, :], time)


def test_analytic_sim_data_gen_3d_scalar_linear() -> None:
    """Test 3D scalar linear field generation and interpolation."""
    sim_data, data_gen = verif.scalar_linear_3d()

    assert sim_data.num_spat_dims == 3
    assert "temperature" in sim_data.node_vars
    n_nodes = sim_data.coords.shape[0]
    assert sim_data.node_vars["temperature"].shape[0] == n_nodes

    # Create Felix FieldScalar
    field = fx.FieldScalar(
        sim_data=sim_data,
        comp_key="temperature",
        spatial_dims=fx.EDim.THREED,
    )

    # Evaluate at off-grid query points
    eval_points = np.array(
        [
            [2.5, 1.8, 1.2],
            [5.0, 3.75, 2.5],
            [7.5, 5.6, 3.8],
        ]
    )

    # Truth from SymPy
    truth_vals = data_gen.evaluate_field_truth(
        field_key="temperature",
        coords=eval_points,
        time_steps=sim_data.time,
    )

    # Sample from Felix field interpolator
    sampled_vals = field.sample_field(
        points=eval_points,
        times=sim_data.time,
    )

    # For a trilinear field on hex elements, interpolation is exact
    np.testing.assert_allclose(sampled_vals[:, 0, :], truth_vals, rtol=1e-5)


def test_analytic_sim_data_gen_3d_vector_linear() -> None:
    """Test 3D vector linear field generation and interpolation."""
    sim_data, data_gen = verif.vector_linear_3d()

    assert sim_data.num_spat_dims == 3
    for k in ("disp_x", "disp_y", "disp_z"):
        assert k in sim_data.node_vars

    field = fx.FieldVector(
        sim_data=sim_data,
        comp_keys=("disp_x", "disp_y", "disp_z"),
        spatial_dims=fx.EDim.THREED,
    )

    eval_points = np.array([[3.0, 2.0, 1.0], [6.0, 4.0, 2.0]])
    sampled = field.sample_field(
        points=eval_points,
        times=sim_data.time,
    )

    truth_all = data_gen.evaluate_all_fields_truth(
        coords=eval_points,
        time_steps=sim_data.time,
    )

    for ii, k in enumerate(("disp_x", "disp_y", "disp_z")):
        np.testing.assert_allclose(sampled[:, ii, :], truth_all[k], rtol=1e-5)


def test_integrate_symbolic_3d() -> None:
    """Test exact SymPy symbolic integration helper on a 3D box."""
    case_data = verif.standard_case_3d(("temp",))
    sym_z, sym_y, sym_x, sym_t = sympy.symbols("z,y,x,t")

    case_data.funcs_x = {"temp": sym_x}
    case_data.funcs_y = {"temp": sym_y}
    case_data.funcs_z = {"temp": sym_z}
    case_data.funcs_t = {"temp": 1.0}

    data_gen = verif.AnalyticSimDataGen(case_data)

    # Exact integral of x*y*z over [0, 2] x [0, 3] x [0, 4]:
    # (2^2/2) * (3^2/2) * (4^2/2) = 2 * 4.5 * 8 = 72.0
    integral = data_gen.integrate_symbolic(
        field_key="temp",
        bounds_x=(0.0, 2.0),
        bounds_y=(0.0, 3.0),
        bounds_z=(0.0, 4.0),
    )

    assert np.isclose(integral, 72.0, rtol=1e-12)
