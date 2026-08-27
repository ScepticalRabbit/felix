# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
import copy
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from felix.sensorsim.enums import EErrType, EErrDep
from felix.sensorsim.errorsimulator import IErrSimulator
from felix.sensorsim.errordriftcalc import IDriftCalculator
from felix.sensorsim.generatorsrandom import IGenRandom
from felix.sensorsim.sensordata import SensorData


@dataclass(slots=True)
class ErrFieldData:
    pos_offset_xyz: np.ndarray | None = None
    ang_offset_zyx: np.ndarray | None = None
    time_offset: np.ndarray | None = None
    pos_rand_xyz: tuple[
        IGenRandom | None, IGenRandom | None, IGenRandom | None
    ] = (None, None, None)
    ang_rand_zyx: tuple[
        IGenRandom | None, IGenRandom | None, IGenRandom | None
    ] = (None, None, None)
    time_rand: IGenRandom | None = None
    spatial_averager: object | None = None
    spatial_dims: np.ndarray | None = None
    pos_lock_xyz: np.ndarray | None = None
    ang_lock_zyx: np.ndarray | None = None
    time_drift: IDriftCalculator | None = None


class ErrSysField(IErrSimulator):
    __slots__ = (
        "_field",
        "_field_err_data",
        "_err_dep",
        "_sensor_data_perturbed",
    )

    def __init__(
        self,
        field: object,
        field_err_data: ErrFieldData,
        err_dep: EErrDep = EErrDep.DEPENDENT,
    ) -> None:
        self._field = field
        self._field_err_data = field_err_data
        self._err_dep = err_dep
        self._sensor_data_perturbed = SensorData()

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def get_error_dep(self) -> EErrDep:
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self._err_dep = dependence

    def get_perturbed_sensor_data(self) -> SensorData:
        return self._sensor_data_perturbed

    def reseed(self, seed: int | None = None) -> None:
        for rr in self._field_err_data.pos_rand_xyz:
            if rr is not None:
                rr.reseed(seed)
        for rr in self._field_err_data.ang_rand_zyx:
            if rr is not None:
                rr.reseed(seed)
        if self._field_err_data.time_rand is not None:
            self._field_err_data.time_rand.reseed(seed)

    def sim_errs(
        self,
        err_basis: np.ndarray,
        sens_data: SensorData,
    ) -> tuple[np.ndarray, SensorData]:
        self._sensor_data_perturbed = copy.deepcopy(sens_data)
        self._sensor_data_perturbed.spatial_averager = (
            self._field_err_data.spatial_averager
        )
        self._sensor_data_perturbed.spatial_dims = (
            self._field_err_data.spatial_dims
        )

        self._sensor_data_perturbed.positions = _perturb_sensor_positions(
            self._sensor_data_perturbed.positions,
            self._field_err_data.pos_offset_xyz,
            self._field_err_data.pos_rand_xyz,
            self._field_err_data.pos_lock_xyz,
        )

        self._sensor_data_perturbed.sample_times = _perturb_sample_times(
            self._field.get_time_steps(),
            self._sensor_data_perturbed.sample_times,
            self._field_err_data.time_offset,
            self._field_err_data.time_rand,
            self._field_err_data.time_drift,
        )

        self._sensor_data_perturbed.angles = _perturb_sensor_angles(
            sens_data.positions.shape[0],
            self._sensor_data_perturbed.angles,
            self._field_err_data.ang_offset_zyx,
            self._field_err_data.ang_rand_zyx,
            self._field_err_data.ang_lock_zyx,
        )

        sampled = self._field.sample_field(
            self._sensor_data_perturbed.positions,
            self._sensor_data_perturbed.sample_times,
            self._sensor_data_perturbed.angles,
        )
        sys_errs = sampled - err_basis
        return (sys_errs, self._sensor_data_perturbed)

    def to_spec_dict(self) -> dict:
        return {}


def _perturb_sensor_positions(
    sens_pos_nominal: np.ndarray,
    pos_offset_xyz: np.ndarray | None,
    pos_rand_xyz: tuple[
        IGenRandom | None, IGenRandom | None, IGenRandom | None
    ] | None,
    pos_lock_xyz: np.ndarray | None,
) -> np.ndarray:
    sens_pos_perturbed = np.copy(sens_pos_nominal)

    if pos_offset_xyz is not None:
        sens_pos_perturbed = sens_pos_perturbed + pos_offset_xyz

    if pos_rand_xyz is not None:
        for ii, rng in enumerate(pos_rand_xyz):
            if rng is not None:
                sens_pos_perturbed[:, ii] = (
                    sens_pos_perturbed[:, ii]
                    + rng.generate(shape=(sens_pos_perturbed.shape[0],))
                )

    if pos_lock_xyz is not None:
        sens_pos_perturbed[pos_lock_xyz] = sens_pos_nominal[pos_lock_xyz]

    return sens_pos_perturbed


def _perturb_sample_times(
    sim_time: np.ndarray,
    time_nominal: np.ndarray | None,
    time_offset: np.ndarray | None,
    time_rand: IGenRandom | None,
    time_drift: IDriftCalculator | None,
) -> np.ndarray | None:
    if time_nominal is None:
        if (
            time_offset is not None
            or time_rand is not None
            or time_drift is not None
        ):
            time_nominal = sim_time
        else:
            return None

    time_perturbed = np.copy(time_nominal)

    if time_offset is not None:
        time_perturbed = time_perturbed + time_offset
    if time_rand is not None:
        rand_shape = time_perturbed.shape
        time_perturbed = time_perturbed + time_rand.generate(shape=rand_shape)
    if time_drift is not None:
        time_perturbed = time_perturbed + time_drift.calc_drift(time_perturbed)

    return time_perturbed


def _perturb_sensor_angles(
    n_sensors: int,
    angles_nominal: tuple[Rotation, ...] | None,
    angle_offsets_zyx: np.ndarray | None,
    rand_ang_zyx: tuple[
        IGenRandom | None, IGenRandom | None, IGenRandom | None
    ] | None,
    angle_lock_zyx: np.ndarray | None,
) -> tuple[Rotation, ...] | None:
    if angles_nominal is None:
        if angle_offsets_zyx is not None or rand_ang_zyx is not None:
            angles_nominal = n_sensors * (
                Rotation.from_euler("zyx", [0, 0, 0], degrees=True),
            )
        else:
            return None

    angles_perturbed = [Rotation.from_euler("zyx", [0, 0, 0], degrees=True)] * len(
        angles_nominal
    )
    for ii, rot_nom in enumerate(angles_nominal):
        sensor_rot_angs = np.zeros((3,))

        if angle_offsets_zyx is not None:
            sensor_rot_angs = sensor_rot_angs + angle_offsets_zyx[ii, :]

        if rand_ang_zyx is not None:
            for jj, rand_ang in enumerate(rand_ang_zyx):
                if rand_ang is not None:
                    sensor_rot_angs[jj] = (
                        sensor_rot_angs[jj]
                        + rand_ang.generate(shape=(1,))[0]
                    )

        if angle_lock_zyx is not None:
            sensor_rot_angs[angle_lock_zyx[ii, :]] = 0.0

        sensor_rot = Rotation.from_euler("zyx", sensor_rot_angs, degrees=True)
        angles_perturbed[ii] = sensor_rot * rot_nom

    return tuple(angles_perturbed)
