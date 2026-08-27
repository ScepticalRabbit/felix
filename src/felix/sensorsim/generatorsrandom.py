# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from abc import ABC, abstractmethod
import numpy as np


class IGenRandom(ABC):
    @abstractmethod
    def generate(self, shape: tuple[int, ...]) -> np.ndarray:
        pass

    @abstractmethod
    def reseed(self, seed: int | None = None) -> None:
        pass

    @abstractmethod
    def to_spec_dict(self) -> dict:
        pass


class GenUniform(IGenRandom):
    __slots__ = ("_low", "_high", "_seed", "_rng")

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._low = low
        self._high = high
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.uniform(self._low, self._high, size=shape)

    def reseed(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def to_spec_dict(self) -> dict:
        return {
            "dist_type": 1,
            "param0": float(self._low),
            "param1": float(self._high),
            "param2": 0.0,
            "seed": self._seed,
        }


class GenNormal(IGenRandom):
    __slots__ = ("_mean", "_std", "_seed", "_rng")

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._mean = mean
        self._std = std
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.normal(self._mean, self._std, size=shape)

    def reseed(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def to_spec_dict(self) -> dict:
        return {
            "dist_type": 2,
            "param0": float(self._mean),
            "param1": float(self._std),
            "param2": 0.0,
            "seed": self._seed,
        }


class GenTriangular(IGenRandom):
    __slots__ = ("_left", "_mode", "_right", "_seed", "_rng")

    def __init__(
        self,
        left: float = -1.0,
        mode: float = 0.0,
        right: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._left = left
        self._mode = mode
        self._right = right
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.triangular(
            self._left, self._mode, self._right, size=shape
        )

    def reseed(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def to_spec_dict(self) -> dict:
        return {
            "dist_type": 3,
            "param0": float(self._left),
            "param1": float(self._mode),
            "param2": float(self._right),
            "seed": self._seed,
        }


class GenExponential(IGenRandom):
    __slots__ = ("_scale", "_seed", "_rng")

    def __init__(
        self,
        scale: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._scale = scale
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.exponential(self._scale, size=shape)

    def reseed(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def to_spec_dict(self) -> dict:
        return {
            "dist_type": 4,
            "param0": float(1.0 / self._scale if self._scale != 0 else 1.0),
            "param1": 0.0,
            "param2": 0.0,
            "seed": self._seed,
        }


class GenGamma(IGenRandom):
    __slots__ = ("_shape", "_scale", "_seed", "_rng")

    def __init__(
        self,
        shape: float = 1.0,
        scale: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._shape = shape
        self._scale = scale
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.gamma(self._shape, self._scale, size=shape)

    def reseed(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def to_spec_dict(self) -> dict:
        return {
            "dist_type": 5,
            "param0": float(self._shape),
            "param1": float(self._scale),
            "param2": 0.0,
            "seed": self._seed,
        }


class GenBeta(IGenRandom):
    __slots__ = ("_a", "_b", "_seed", "_rng")

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._a = a
        self._b = b
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.beta(self._a, self._b, size=shape)

    def reseed(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def to_spec_dict(self) -> dict:
        return {
            "dist_type": 6,
            "param0": float(self._a),
            "param1": float(self._b),
            "param2": 0.0,
            "seed": self._seed,
        }


class GenLogNormal(IGenRandom):
    __slots__ = ("_mean", "_sigma", "_seed", "_rng")

    def __init__(
        self,
        mean: float = 0.0,
        sigma: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._mean = mean
        self._sigma = sigma
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def generate(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.lognormal(self._mean, self._sigma, size=shape)

    def reseed(self, seed: int | None = None) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def to_spec_dict(self) -> dict:
        return {
            "dist_type": 7,
            "param0": float(self._mean),
            "param1": float(self._sigma),
            "param2": 0.0,
            "seed": self._seed,
        }
