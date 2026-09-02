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
from pyvale.dataio.simdata import SimData

import felix as fx


@pytest.fixture
def sample_sim_data_2d() -> SimData:
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    connect = {"quad4": np.array([[0, 1, 2, 3]], dtype=np.int64)}
    node_vars = {
        "temp": np.array([
            [100.0, 105.0, 110.0],
            [120.0, 125.0, 130.0],
            [140.0, 145.0, 150.0],
            [110.0, 115.0, 120.0],
        ], dtype=np.float64)
    }
    return SimData(
        num_spat_dims=2,
        time=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        coords=coords,
        connect=connect,
        node_vars=node_vars,
    )


def test_experiment_simulator_multithreading_parity(
    sample_sim_data_2d: SimData,
) -> None:
    field = fx.FieldScalar(sample_sim_data_2d, "temp", fx.EDim.TWOD)
    positions = np.array([
        [0.25, 0.25, 0.0],
        [0.75, 0.25, 0.0],
        [0.75, 0.75, 0.0],
        [0.25, 0.75, 0.0],
    ], dtype=np.float64)
    sens_data = fx.SensorData(positions=positions)
    sensors = fx.SensorsPoint(sens_data, field)

    err_chain = [
        fx.ErrSysOffset(offset=1.5),
        fx.ErrRandGen(
            gen_rand=fx.GenNormal(mean=0.0, std=0.25, seed=12345),
            err_dep=fx.EErrDep.INDEPENDENT,
        ),
    ]
    sensors.set_error_chain(err_chain)

    num_experiments = 32

    opts_seq = fx.ExpSimOpts(
        num_experiments=num_experiments,
        num_threads=1,
        seed=42,
    )
    exp_sim_seq = fx.ExperimentSimulator(sensors, opts=opts_seq)
    meas_seq = exp_sim_seq.sim_experiments()

    opts_par = fx.ExpSimOpts(
        num_experiments=num_experiments,
        num_threads=4,
        grain_size=2,
        seed=42,
    )
    exp_sim_par = fx.ExperimentSimulator(sensors, opts=opts_par)
    meas_par = exp_sim_par.sim_experiments()

    assert meas_seq.shape == (num_experiments, 4, 1, 3)
    assert meas_par.shape == (num_experiments, 4, 1, 3)
    assert np.allclose(meas_seq, meas_par, rtol=1e-12, atol=1e-12)


def test_sensors_point_sim_experiments_multithreaded(
    sample_sim_data_2d: SimData,
) -> None:
    field = fx.FieldScalar(sample_sim_data_2d, "temp", fx.EDim.TWOD)
    positions = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)
    sens_data = fx.SensorData(positions=positions)
    sensors = fx.SensorsPoint(sens_data, field)

    truth, meas, _, _, _, _, _ = sensors.sim_experiments(
        num_experiments=16,
        num_threads=2,
        seed=999,
    )
    assert truth.shape == (16, 1, 1, 3)
    assert meas.shape == (16, 1, 1, 3)
