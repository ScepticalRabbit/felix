# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from dataclasses import dataclass

import numpy as np

from felix.cython import felix as fc


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
    data: dict[tuple, np.ndarray] | np.ndarray,
) -> dict[tuple, ExpSimStats] | ExpSimStats:
    if isinstance(data, dict):
        return {
            key: _calc_single_stats(values)
            for key, values in data.items()
            if isinstance(values, np.ndarray) and values.ndim >= 3
        }
    return _calc_single_stats(data)


def _calc_single_stats(values: np.ndarray) -> ExpSimStats:
    result = fc.calc_experiment_stats(values)
    return ExpSimStats(*result)
