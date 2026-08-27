# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import numpy as np
import pyvale.verif.analyticsimdatafactory as analytic

import felix


POSITIONS_2D = np.array(
    [[2.5, 1.8, 0.0], [5.0, 3.75, 0.0], [7.5, 5.6, 0.0]],
    dtype=np.float64,
)
POSITIONS_3D = np.array(
    [[2.5, 1.8, 1.2], [5.0, 3.75, 2.5], [7.5, 5.6, 3.8]],
    dtype=np.float64,
)
SAMPLE_TIMES = np.array([0.2, 0.7], dtype=np.float64)


def test_scalar_2d_matches_analytic_truth() -> None:
    sim_data, data_gen = analytic.scalar_linear_2d()
    field = felix.FieldScalar(sim_data, "temperature", felix.EDim.TWOD)
    actual = field.sample_field(POSITIONS_2D, SAMPLE_TIMES)
    expected = data_gen.evaluate_field_truth(
        "temperature",
        POSITIONS_2D,
        SAMPLE_TIMES,
    )
    assert np.allclose(actual[:, 0, :], expected)


def test_vector_2d_matches_analytic_truth() -> None:
    sim_data, data_gen = analytic.vector_linear_2d()
    components = ("disp_x", "disp_y")
    field = felix.FieldVector(sim_data, components, felix.EDim.TWOD)
    actual = field.sample_field(POSITIONS_2D, SAMPLE_TIMES)
    expected = data_gen.evaluate_all_fields_truth(POSITIONS_2D, SAMPLE_TIMES)
    for comp, key in enumerate(components):
        assert np.allclose(actual[:, comp, :], expected[key])


def test_tensor_2d_matches_analytic_truth() -> None:
    sim_data, data_gen = analytic.tensor_linear_2d()
    normal = ("strain_xx", "strain_yy")
    deviatoric = ("strain_xy",)
    field = felix.FieldTensor(
        sim_data,
        normal,
        deviatoric,
        felix.EDim.TWOD,
    )
    actual = field.sample_field(POSITIONS_2D, SAMPLE_TIMES)
    expected = data_gen.evaluate_all_fields_truth(POSITIONS_2D, SAMPLE_TIMES)
    for comp, key in enumerate(normal + deviatoric):
        assert np.allclose(actual[:, comp, :], expected[key])


def test_scalar_3d_matches_analytic_truth() -> None:
    sim_data, data_gen = analytic.scalar_linear_3d()
    field = felix.FieldScalar(sim_data, "temperature", felix.EDim.THREED)
    actual = field.sample_field(POSITIONS_3D, SAMPLE_TIMES)
    expected = data_gen.evaluate_field_truth(
        "temperature",
        POSITIONS_3D,
        SAMPLE_TIMES,
    )
    assert np.allclose(actual[:, 0, :], expected)


def test_vector_3d_matches_analytic_truth() -> None:
    sim_data, data_gen = analytic.vector_linear_3d()
    components = ("disp_x", "disp_y", "disp_z")
    field = felix.FieldVector(sim_data, components, felix.EDim.THREED)
    actual = field.sample_field(POSITIONS_3D, SAMPLE_TIMES)
    expected = data_gen.evaluate_all_fields_truth(POSITIONS_3D, SAMPLE_TIMES)
    for comp, key in enumerate(components):
        assert np.allclose(actual[:, comp, :], expected[key])


def test_tensor_3d_matches_analytic_truth() -> None:
    sim_data, data_gen = analytic.tensor_linear_3d()
    normal = ("strain_xx", "strain_yy", "strain_zz")
    deviatoric = ("strain_xy", "strain_yz", "strain_xz")
    field = felix.FieldTensor(
        sim_data,
        normal,
        deviatoric,
        felix.EDim.THREED,
    )
    actual = field.sample_field(POSITIONS_3D, SAMPLE_TIMES)
    expected = data_gen.evaluate_all_fields_truth(POSITIONS_3D, SAMPLE_TIMES)
    for comp, key in enumerate(normal + deviatoric):
        assert np.allclose(actual[:, comp, :], expected[key])
