# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from pyvale.dataio.simdata import SimData

from felix.python.sensorspoint import SensorsPoint


class EExpSimPara(Enum):
    ALL = auto()
    SPLIT = auto()


@dataclass(slots=True)
class ExpSimSaveKeys:
    meas: str = "meas"
    sens_times: str = "sens_times"
    sys: str | None = "sys_errs"
    rand: str | None = "rand_errs"
    truth: str | None = None
    pert_sens_times: str | None = "pert_sens_times"
    pert_sens_pos: str | None = "pert_sens_pos"


@dataclass(slots=True)
class ExpSimOpts:
    workers: int = 1
    num_threads: int = 1
    grain_size: int = 1
    seed_stride: int = 1000
    para: EExpSimPara = EExpSimPara.ALL
    save_keys: ExpSimSaveKeys = field(default_factory=ExpSimSaveKeys)
    num_experiments: int = 100
    seed: int | None = None


class ExperimentSimulator:
    """Thin multi-configuration wrapper over Zig experiment batches."""

    __slots__ = ("_sims", "_sensors", "_opts", "_results", "_pool")

    def __init__(
        self,
        sims: dict[str, SimData] | SensorsPoint,
        sensors: dict[str, SensorsPoint] | None = None,
        opts: ExpSimOpts | None = None,
        save_keys: ExpSimSaveKeys | None = None,
    ) -> None:
        if isinstance(sims, SensorsPoint):
            self._sims = {"sim_0": sims.get_field().get_sim_data()}
            self._sensors = {"sens_0": sims}
        else:
            self._sims = sims
            self._sensors = sensors if sensors is not None else {}
        self._opts = opts if opts is not None else ExpSimOpts()
        if save_keys is not None:
            self._opts.save_keys = save_keys
        self._results = None

        threads = (
            self._opts.num_threads
            if self._opts.num_threads != 1
            else self._opts.workers
        )
        if threads > 1 or threads == 0:
            from felix.cython.felix import PyFelixThreadPool

            self._pool = PyFelixThreadPool(threads)
        else:
            self._pool = None

    def run_experiments(
        self,
        num_exp_per_sim: int = 100,
    ) -> dict[tuple[str, str, str], np.ndarray]:
        results: dict[tuple[str, str, str], np.ndarray] = {}
        seed = 0 if self._opts.seed is None else self._opts.seed
        threads = (
            self._opts.num_threads
            if self._opts.num_threads != 1
            else self._opts.workers
        )
        for sim_key, sim_data in self._sims.items():
            for sens_key, sensors in self._sensors.items():
                sensors.get_field().set_sim_data(sim_data)
                truth, meas, sys, rand, _, pert_pos, pert_times = (
                    sensors.sim_experiments(
                        num_exp_per_sim,
                        seed=seed,
                        num_threads=threads,
                        seed_stride=self._opts.seed_stride,
                        grain_size=self._opts.grain_size,
                        thread_pool=self._pool,
                    )
                )
                keys = self._opts.save_keys
                if keys.meas:
                    results[(sim_key, sens_key, keys.meas)] = meas
                if keys.sens_times:
                    results[(sim_key, sens_key, keys.sens_times)] = (
                        sensors.get_sample_times()
                    )
                if keys.sys:
                    results[(sim_key, sens_key, keys.sys)] = sys
                if keys.rand:
                    results[(sim_key, sens_key, keys.rand)] = rand
                if keys.truth:
                    results[(sim_key, sens_key, keys.truth)] = truth[0]
                if keys.pert_sens_pos:
                    results[(sim_key, sens_key, keys.pert_sens_pos)] = pert_pos
                if keys.pert_sens_times:
                    results[(sim_key, sens_key, keys.pert_sens_times)] = (
                        pert_times
                    )
        self._results = results
        return results

    def sim_experiments(
        self,
        num_experiments: int | None = None,
    ) -> np.ndarray:
        count = (
            self._opts.num_experiments
            if num_experiments is None
            else num_experiments
        )
        results = self.run_experiments(count)
        first = next(key for key in results if key[2] == "meas")
        return results[first]
