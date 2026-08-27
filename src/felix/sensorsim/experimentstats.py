# ==============================================================================
# Felix: A High Performance Sensor Simulation Core
# License: MIT
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class ExpSimStats:
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    median: np.ndarray
    var: np.ndarray
    mad: np.ndarray


def calc_exp_sim_stats(
    data: dict[tuple, np.ndarray] | np.ndarray
) -> dict[tuple, ExpSimStats] | ExpSimStats:
    """Calculate summary statistics over the experiment dimension (axis 0)."""
    if isinstance(data, dict):
        stats_dict = {}
        for key, arr in data.items():
            if isinstance(arr, np.ndarray) and arr.ndim >= 3:
                stats_dict[key] = calc_single_stats(arr)
        return stats_dict
    return calc_single_stats(data)


def calc_single_stats(measurements: np.ndarray) -> ExpSimStats:
    mean_val = np.mean(measurements, axis=0)
    std_val = np.std(measurements, axis=0)
    min_val = np.min(measurements, axis=0)
    max_val = np.max(measurements, axis=0)
    median_val = np.median(measurements, axis=0)
    var_val = np.var(measurements, axis=0)
    mad_val = np.median(np.abs(measurements - median_val[None, ...]), axis=0)

    return ExpSimStats(
        mean=mean_val,
        std=std_val,
        min=min_val,
        max=max_val,
        median=median_val,
        var=var_val,
        mad=mad_val,
    )
