# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from felix.python.enums import EErrDep, EErrType, ERoundMethod
from felix.python.fieldspecs import FieldSpec


class IGenRandom:
    """Compatibility interface for a Zig random-distribution specification."""

    def to_spec_dict(self) -> dict:
        raise NotImplementedError

    def reseed(self, seed: int | None = None) -> None:
        self.seed = seed


@dataclass(slots=True)
class GenUniform(IGenRandom):
    low: float = 0.0
    high: float = 1.0
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        return _gen_spec(1, self.low, self.high, 0.0, self.seed)


@dataclass(slots=True)
class GenNormal(IGenRandom):
    mean: float = 0.0
    std: float = 1.0
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        return _gen_spec(2, self.mean, self.std, 0.0, self.seed)


@dataclass(slots=True)
class GenTriangular(IGenRandom):
    left: float = -1.0
    mode: float = 0.0
    right: float = 1.0
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        return _gen_spec(3, self.left, self.mode, self.right, self.seed)


@dataclass(slots=True)
class GenExponential(IGenRandom):
    scale: float = 1.0
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        rate = 1.0 / self.scale if self.scale != 0.0 else 1.0
        return _gen_spec(4, rate, 0.0, 0.0, self.seed)


@dataclass(slots=True)
class GenGamma(IGenRandom):
    shape: float = 1.0
    scale: float = 1.0
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        return _gen_spec(5, self.shape, self.scale, 0.0, self.seed)


@dataclass(slots=True)
class GenBeta(IGenRandom):
    a: float = 1.0
    b: float = 1.0
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        return _gen_spec(6, self.a, self.b, 0.0, self.seed)


@dataclass(slots=True)
class GenLogNormal(IGenRandom):
    mean: float = 0.0
    sigma: float = 1.0
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        return _gen_spec(7, self.mean, self.sigma, 0.0, self.seed)


@dataclass(slots=True)
class GenChiSquare(IGenRandom):
    dofs: float = 1.0
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        return _gen_spec(8, self.dofs, 0.0, 0.0, self.seed)


@dataclass(slots=True)
class GenDirichlet(IGenRandom):
    alpha: tuple[float, ...] = (1.0, 1.0)
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        return _gen_spec(9, 0.0, 0.0, 0.0, self.seed)


@dataclass(slots=True)
class GenF(IGenRandom):
    dofs: tuple[float, float] = (1.0, 1.0)
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        return _gen_spec(10, self.dofs[0], self.dofs[1], 0.0, self.seed)


@dataclass(slots=True)
class GenStandardT(IGenRandom):
    dofs: float = 1.0
    seed: int | None = None

    def to_spec_dict(self) -> dict:
        return _gen_spec(11, self.dofs, 0.0, 0.0, self.seed)


class IErrSimulator:
    """Compatibility adapter for a Zig error specification."""

    err_dep: EErrDep

    def get_error_dep(self) -> EErrDep:
        return self.err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        self.err_dep = dependence

    def reseed(self, seed: int | None = None) -> None:
        generator = getattr(self, "generator", None)
        if generator is not None:
            generator.reseed(seed)

    def to_spec_dict(self) -> dict:
        raise NotImplementedError


@dataclass(slots=True)
class ErrSysOffset(IErrSimulator):
    offset: float
    err_dep: EErrDep = EErrDep.INDEPENDENT

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def to_spec_dict(self) -> dict:
        return _error_spec(0, EErrType.SYSTEMATIC, self.err_dep, self.offset)


@dataclass(slots=True)
class ErrSysOffsetPercent(IErrSimulator):
    offset_percent: float
    err_dep: EErrDep = EErrDep.INDEPENDENT

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def to_spec_dict(self) -> dict:
        return _error_spec(
            1, EErrType.SYSTEMATIC, self.err_dep, self.offset_percent
        )


@dataclass(slots=True, init=False)
class ErrSysGen(IErrSimulator):
    generator: IGenRandom
    err_dep: EErrDep

    def __init__(
        self,
        generator: IGenRandom | None = None,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
        gen: IGenRandom | None = None,
        gen_rand: IGenRandom | None = None,
    ) -> None:
        g = generator or gen or gen_rand
        if g is None:
            raise ValueError("A random generator must be provided")
        self.generator = g
        self.err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def to_spec_dict(self) -> dict:
        return _generator_error_spec(2, EErrType.SYSTEMATIC, self)


@dataclass(slots=True, init=False)
class ErrSysGenPercent(IErrSimulator):
    generator: IGenRandom
    err_dep: EErrDep

    def __init__(
        self,
        generator: IGenRandom | None = None,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
        gen: IGenRandom | None = None,
        gen_rand: IGenRandom | None = None,
    ) -> None:
        g = generator or gen or gen_rand
        if g is None:
            raise ValueError("A random generator must be provided")
        self.generator = g
        self.err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def to_spec_dict(self) -> dict:
        return _generator_error_spec(3, EErrType.SYSTEMATIC, self)


@dataclass(slots=True, init=False)
class ErrRandGen(IErrSimulator):
    generator: IGenRandom
    err_dep: EErrDep

    def __init__(
        self,
        generator: IGenRandom | None = None,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
        gen: IGenRandom | None = None,
        gen_rand: IGenRandom | None = None,
    ) -> None:
        g = generator or gen or gen_rand
        if g is None:
            raise ValueError("A random generator must be provided")
        self.generator = g
        self.err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.RANDOM

    def to_spec_dict(self) -> dict:
        return _generator_error_spec(4, EErrType.RANDOM, self)


@dataclass(slots=True, init=False)
class ErrRandGenPercent(IErrSimulator):
    generator: IGenRandom
    err_dep: EErrDep

    def __init__(
        self,
        generator: IGenRandom | None = None,
        err_dep: EErrDep = EErrDep.INDEPENDENT,
        gen: IGenRandom | None = None,
        gen_rand: IGenRandom | None = None,
    ) -> None:
        g = generator or gen or gen_rand
        if g is None:
            raise ValueError("A random generator must be provided")
        self.generator = g
        self.err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.RANDOM

    def to_spec_dict(self) -> dict:
        return _generator_error_spec(5, EErrType.RANDOM, self)


@dataclass(slots=True, init=False)
class ErrSysRoundOff(IErrSimulator):
    method: ERoundMethod
    base: float
    err_dep: EErrDep

    def __init__(
        self,
        method: ERoundMethod = ERoundMethod.ROUND,
        base: float = 1.0,
        err_dep: EErrDep = EErrDep.DEPENDENT,
    ) -> None:
        self.method = method
        self.base = base
        self.err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def to_spec_dict(self) -> dict:
        return _error_spec(
            6,
            EErrType.SYSTEMATIC,
            self.err_dep,
            self.base,
            _round_value(self.method),
        )


@dataclass(slots=True, init=False)
class ErrSysDigitisation(IErrSimulator):
    units_per_bit: float
    method: ERoundMethod
    err_dep: EErrDep

    def __init__(
        self,
        bits_per_unit: float | None = None,
        units_per_bit: float | None = None,
        method: ERoundMethod = ERoundMethod.ROUND,
        err_dep: EErrDep = EErrDep.DEPENDENT,
    ) -> None:
        if bits_per_unit is not None:
            self.units_per_bit = 1.0 / bits_per_unit
        elif units_per_bit is not None:
            self.units_per_bit = units_per_bit
        else:
            self.units_per_bit = 1.0
        self.method = method
        self.err_dep = err_dep

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def to_spec_dict(self) -> dict:
        return _error_spec(
            7,
            EErrType.SYSTEMATIC,
            self.err_dep,
            self.units_per_bit,
            _round_value(self.method),
        )


@dataclass(slots=True)
class ErrSysSaturation(IErrSimulator):
    meas_min: float = -np.inf
    meas_max: float = np.inf
    err_dep: EErrDep = EErrDep.DEPENDENT

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def to_spec_dict(self) -> dict:
        return _error_spec(
            8,
            EErrType.SYSTEMATIC,
            self.err_dep,
            self.meas_min,
            self.meas_max,
        )


@dataclass(slots=True)
class ErrSysCalibration(IErrSimulator):
    assumed_calib: Callable[[np.ndarray], np.ndarray]
    truth_calib: Callable[[np.ndarray], np.ndarray]
    cal_range: tuple[float, float]
    n_cal_divs: int = 10000
    use_newton: bool = False
    truth_calib_prime: Callable[[np.ndarray], np.ndarray] | None = None
    tol: float = 1e-8
    max_iter: int = 50
    err_dep: EErrDep = EErrDep.INDEPENDENT
    _table: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        raw = np.linspace(*self.cal_range, self.n_cal_divs)
        truth = self.truth_calib(raw)
        assumed = self.assumed_calib(raw)
        order = np.argsort(truth)
        self._table = np.column_stack((truth[order], assumed[order]))

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def to_spec_dict(self) -> dict:
        spec = _error_spec(9, EErrType.SYSTEMATIC, self.err_dep)
        spec["table"] = self._table
        return spec


@dataclass(slots=True)
class DriftConstant(IErrSimulator):
    offset: float = 0.0
    err_dep: EErrDep = EErrDep.INDEPENDENT

    def to_spec_dict(self) -> dict:
        return _error_spec(
            0, EErrType.SYSTEMATIC, self.err_dep, self.offset
        )

    def to_drift_spec_dict(self) -> dict:
        return {"drift_kind": 1, "param0": self.offset}


@dataclass(slots=True)
class DriftLinear(IErrSimulator):
    rate: float
    time_start: float = 0.0
    offset: float = 0.0
    err_dep: EErrDep = EErrDep.INDEPENDENT

    def to_spec_dict(self) -> dict:
        return _error_spec(
            10,
            EErrType.SYSTEMATIC,
            self.err_dep,
            self.rate,
            self.time_start,
            self.offset,
        )

    def to_drift_spec_dict(self) -> dict:
        return {
            "drift_kind": 2,
            "param0": self.rate,
            "param1": self.time_start,
            "param2": self.offset,
        }


@dataclass(slots=True)
class DriftPolynomial(IErrSimulator):
    coeffs: tuple[float, ...]
    time_start: float = 0.0
    err_dep: EErrDep = EErrDep.INDEPENDENT

    def to_spec_dict(self) -> dict:
        spec = _error_spec(
            11,
            EErrType.SYSTEMATIC,
            self.err_dep,
            self.time_start,
        )
        spec["poly_coeffs"] = np.asarray(self.coeffs, dtype=np.float64)
        return spec

    def to_drift_spec_dict(self) -> dict:
        return {
            "drift_kind": 3,
            "param0": self.time_start,
            "poly_coeffs": np.asarray(self.coeffs, dtype=np.float64),
        }


IDriftCalculator = DriftConstant | DriftLinear | DriftPolynomial


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


@dataclass(slots=True)
class ErrSysField(IErrSimulator):
    field: FieldSpec
    field_err_data: ErrFieldData
    err_dep: EErrDep = EErrDep.DEPENDENT

    def get_error_type(self) -> EErrType:
        return EErrType.SYSTEMATIC

    def to_spec_dict(self) -> dict:
        return {
            "kind": 12,
            "err_type": 0,
            "err_dep": _dependence_value(self.err_dep),
            "field_data": self.field_err_data,
        }


@dataclass(slots=True)
class ErrIntOpts:
    force_dependence: EErrDep | None = None
    store_all_errs: bool = False


def _gen_spec(
    dist_type: int,
    param0: float,
    param1: float,
    param2: float,
    seed: int | None,
) -> dict:
    return {
        "dist_type": dist_type,
        "param0": float(param0),
        "param1": float(param1),
        "param2": float(param2),
        "seed": seed,
    }


def _error_spec(
    kind: int,
    err_type: EErrType,
    err_dep: EErrDep,
    param0: float = 0.0,
    param1: float = 0.0,
    param2: float = 0.0,
) -> dict:
    return {
        "kind": kind,
        "err_type": 1 if err_type == EErrType.RANDOM else 0,
        "err_dep": _dependence_value(err_dep),
        "dist_type": 0,
        "param0": float(param0),
        "param1": float(param1),
        "param2": float(param2),
    }


def _generator_error_spec(
    kind: int,
    err_type: EErrType,
    error: ErrSysGen | ErrSysGenPercent | ErrRandGen | ErrRandGenPercent,
) -> dict:
    spec = error.generator.to_spec_dict()
    spec.update(
        kind=kind,
        err_type=1 if err_type == EErrType.RANDOM else 0,
        err_dep=_dependence_value(error.err_dep),
    )
    return spec


def _dependence_value(err_dep: EErrDep) -> int:
    return 1 if err_dep == EErrDep.DEPENDENT else 0


def _round_value(method: ERoundMethod) -> float:
    if method == ERoundMethod.FLOOR:
        return 1.0
    if method == ERoundMethod.CEIL:
        return 2.0
    return 0.0
