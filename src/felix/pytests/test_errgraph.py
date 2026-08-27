# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
"""Tests for Directed Acyclic Graph (DAG) sensor error integration in Felix."""

import numpy as np
import pytest

import felix as fx
import pyvale.verif.analyticsimdatafactory as asd


def test_err_graph_linear_equivalence() -> None:
    """A linear ErrGraph produces identical results to ErrIntegrator."""
    meas_shape = (4, 1, 10)
    seed = 12345
    truth = np.ones(meas_shape) * 10.0

    sens_data = fx.SensorData(
        positions=np.zeros((4, 3)),
        sample_times=np.linspace(0.0, 1.0, 10),
    )

    errs_chain = [
        fx.ErrSysOffset(offset=2.0),
        fx.ErrRandGen(fx.GenNormal(std=0.5, seed=seed)),
        fx.ErrSysSaturation(meas_min=0.0, meas_max=11.0),
    ]

    integrator = fx.ErrIntegrator(
        err_chain=errs_chain,
        sensor_data_initial=sens_data,
        meas_shape=meas_shape,
    )
    integrator.reseed_error_chain(seed)
    chain_total = integrator.calc_errors_from_chain(truth)
    chain_sys = integrator.get_errs_systematic()
    chain_rand = integrator.get_errs_random()

    graph = fx.err_chain_to_graph(
        err_chain=errs_chain,
        meas_shape=meas_shape,
        sensor_data_initial=sens_data,
    )
    graph.reseed(seed)
    graph_total = graph.calc_errors_from_graph(truth)
    graph_sys = graph.get_errs_systematic()
    graph_rand = graph.get_errs_random()

    assert np.allclose(chain_total, graph_total)
    assert np.allclose(chain_sys, graph_sys)
    assert np.allclose(chain_rand, graph_rand)


def test_err_graph_cycle_detection() -> None:
    """ErrGraph raises ValueError when given cyclic node dependencies."""
    meas_shape = (2, 1, 5)
    sens_data = fx.SensorData(positions=np.zeros((2, 3)))

    nodes = [
        fx.ErrNode("node_a", fx.ErrSysOffset(1.0), inputs=("node_c",)),
        fx.ErrNode("node_b", fx.ErrSysOffset(2.0), inputs=("node_a",)),
        fx.ErrNode("node_c", fx.ErrSysOffset(3.0), inputs=("node_b",)),
    ]

    with pytest.raises(ValueError, match="cycle"):
        fx.ErrGraph(nodes, meas_shape, sens_data)


def test_err_graph_missing_input_raises() -> None:
    """ErrGraph raises KeyError when an input node name does not exist."""
    meas_shape = (2, 1, 5)
    sens_data = fx.SensorData(positions=np.zeros((2, 3)))

    nodes = [
        fx.ErrNode(
            "node_a", fx.ErrSysOffset(1.0), inputs=("non_existent",)
        ),
    ]

    with pytest.raises(KeyError, match="non_existent"):
        fx.ErrGraph(nodes, meas_shape, sens_data)


def test_err_graph_duplicate_names_raises() -> None:
    """ErrGraph raises ValueError when node names are duplicated."""
    meas_shape = (2, 1, 5)
    sens_data = fx.SensorData(positions=np.zeros((2, 3)))

    nodes = [
        fx.ErrNode("duplicate", fx.ErrSysOffset(1.0)),
        fx.ErrNode("duplicate", fx.ErrSysOffset(2.0)),
    ]

    with pytest.raises(ValueError, match="unique"):
        fx.ErrGraph(nodes, meas_shape, sens_data)


def test_err_graph_diamond_dag() -> None:
    """ErrGraph evaluates diamond DAG dependencies with multiple branches."""
    meas_shape = (2, 1, 5)
    truth = np.full(meas_shape, 5.0)
    sens_data = fx.SensorData(positions=np.zeros((2, 3)))

    builder = fx.ErrGraphBuilder()
    (
        builder.add_root("root_offset", fx.ErrSysOffset(offset=2.0))
        .add_child(
            "branch_a", fx.ErrSysOffset(offset=1.0), parent="root_offset"
        )
        .add_child(
            "branch_b", fx.ErrSysOffset(offset=3.0), parent="root_offset"
        )
        .add_node(
            "sink_sat",
            fx.ErrSysSaturation(meas_min=0.0, meas_max=10.0),
            inputs=("branch_a", "branch_b"),
        )
    )

    opts = fx.ErrGraphOpts(store_node_outputs=True)
    graph = builder.build(meas_shape, sens_data, opts=opts)
    total_err = graph.calc_errors_from_graph(truth)

    assert np.allclose(truth + total_err, 10.0)

    node_outputs = graph.get_node_outputs()
    assert node_outputs is not None
    assert "branch_a" in node_outputs
    assert np.allclose(node_outputs["branch_a"].values, 8.0)
    assert np.allclose(node_outputs["branch_b"].values, 10.0)


def test_err_graph_multiplication_and_replace() -> None:
    """ErrGraph supports EErrOp.MULTIPLY and EErrOp.REPLACE."""
    meas_shape = (2, 1, 4)
    truth = np.full(meas_shape, 10.0)
    sens_data = fx.SensorData(positions=np.zeros((2, 3)))

    nodes = [
        fx.ErrNode(
            "gain_error",
            fx.ErrSysOffset(offset=0.1),
            inputs=(),
            op=fx.EErrOp.MULTIPLY,
        ),
        fx.ErrNode(
            "clipper",
            fx.ErrSysSaturation(meas_min=0.0, meas_max=10.5),
            inputs=("gain_error",),
            op=fx.EErrOp.ADD,
        ),
    ]

    graph = fx.ErrGraph(nodes, meas_shape, sens_data)
    total_err = graph.calc_errors_from_graph(truth)

    assert np.allclose(truth + total_err, 10.5)


def test_sensors_point_with_err_graph() -> None:
    """SensorsPoint integrates cleanly with an ErrGraph."""
    sim_data, _ = asd.scalar_linear_2d()
    field = fx.FieldScalar(
        sim_data,
        comp_key="temperature",
        spatial_dims=fx.EDim.TWOD,
    )

    sens_pos = np.array([[2.0, 2.0, 0.0]])
    sens_data = fx.SensorData(
        positions=sens_pos,
        sample_times=np.array([0.0]),
    )
    sensors = fx.SensorsPoint(sens_data, field)

    builder = fx.ErrGraphBuilder()
    builder.add_root("offset", fx.ErrSysOffset(offset=5.0))
    graph = builder.build(sensors.get_measurement_shape(), sens_data)

    sensors.set_error_graph(graph)
    truth = sensors.get_truth()
    meas = sensors.sim_measurements()

    assert np.allclose(meas, truth + 5.0)
    assert np.allclose(sensors.get_errors_systematic(), 5.0)
    assert np.allclose(sensors.get_errors_random(), 0.0)
