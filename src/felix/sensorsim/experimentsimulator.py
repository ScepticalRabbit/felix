# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import copy
from dataclasses import dataclass, field
import enum
import numpy as np
from pyvale.dataio.simdata import SimData
from felix.sensorsim.sensorspoint import SensorsPoint


class EExpSimPara(enum.Enum):
    ALL = enum.auto()
    SPLIT = enum.auto()


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
    para: EExpSimPara = EExpSimPara.ALL
    save_keys: ExpSimSaveKeys = field(default_factory=ExpSimSaveKeys)
    num_experiments: int = 100
    seed: int | None = None


class ExperimentSimulator:
    __slots__ = ("_sims", "_sensors", "_opts", "_results")

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

    def run_experiments(
        self, num_exp_per_sim: int = 100
    ) -> dict[tuple[str, str, str], np.ndarray]:
        results = {}

        for sim_key, sim_data in self._sims.items():
            for sens_key, sens_array in self._sensors.items():
                sens_copy = copy.deepcopy(sens_array)
                sens_copy.get_field().set_sim_data(sim_data)

                meas_shape = sens_copy.get_measurement_shape()
                n_sensors = meas_shape[0]
                n_times = meas_shape[2]

                meas_arr = np.zeros(
                    (num_exp_per_sim,) + meas_shape, dtype=np.float64
                )
                sys_arr = np.zeros(
                    (num_exp_per_sim,) + meas_shape, dtype=np.float64
                )
                rand_arr = np.zeros(
                    (num_exp_per_sim,) + meas_shape, dtype=np.float64
                )
                pert_pos_arr = np.zeros(
                    (num_exp_per_sim, n_sensors, 3), dtype=np.float64
                )
                pert_times_arr = np.zeros(
                    (num_exp_per_sim, n_times), dtype=np.float64
                )

                err_int = sens_copy.get_error_integrator()
                if self._opts.seed is not None and err_int is not None:
                    err_int.reseed_error_chain(self._opts.seed)

                for ee in range(num_exp_per_sim):
                    meas_arr[ee] = sens_copy.sim_measurements()
                    if err_int is not None:
                        sys_arr[ee] = err_int.get_errs_systematic()
                        rand_arr[ee] = err_int.get_errs_random()

                    pert_sens = sens_copy.get_sensor_data_perturbed()
                    if pert_sens.positions is not None:
                        pert_pos_arr[ee] = pert_sens.positions
                    if pert_sens.sample_times is not None:
                        pert_times_arr[ee] = pert_sens.sample_times

                save_keys = self._opts.save_keys
                if save_keys.meas:
                    results[(sim_key, sens_key, save_keys.meas)] = meas_arr
                if save_keys.sens_times:
                    results[(sim_key, sens_key, save_keys.sens_times)] = (
                        sens_copy.get_sample_times()
                    )
                if save_keys.sys:
                    results[(sim_key, sens_key, save_keys.sys)] = sys_arr
                if save_keys.rand:
                    results[(sim_key, sens_key, save_keys.rand)] = rand_arr
                if save_keys.truth:
                    results[(sim_key, sens_key, save_keys.truth)] = (
                        sens_copy.get_truth()
                    )
                if save_keys.pert_sens_pos:
                    results[(sim_key, sens_key, save_keys.pert_sens_pos)] = (
                        pert_pos_arr
                    )
                if save_keys.pert_sens_times:
                    results[(sim_key, sens_key, save_keys.pert_sens_times)] = (
                        pert_times_arr
                    )

        self._results = results
        return results

    def sim_experiments(
        self, num_experiments: int | None = None
    ) -> np.ndarray:
        n_exp = (
            num_experiments
            if num_experiments is not None
            else self._opts.num_experiments
        )
        res_dict = self.run_experiments(num_exp_per_sim=n_exp)
        first_key = next(k for k in res_dict.keys() if k[2] == "meas")
        return res_dict[first_key]
