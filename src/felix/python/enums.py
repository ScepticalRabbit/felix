# ==============================================================================
# Felix: A virtual sensor laboratory
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# ==============================================================================
from enum import Enum, auto


class EDim(Enum):
    TWOD = 2
    THREED = 3


class EErrType(Enum):
    SYSTEMATIC = auto()
    RANDOM = auto()


class EErrDep(Enum):
    INDEPENDENT = auto()
    DEPENDENT = auto()


class ERoundMethod(Enum):
    ROUND = auto()
    FLOOR = auto()
    CEIL = auto()


class EIntegrationMode(Enum):
    AVERAGE = auto()
    ACCUMULATE = auto()


class EDifferentialMode(Enum):
    STRAIN = auto()
    DIFFERENCE = auto()
    RATIO = auto()
    CUSTOM = auto()


class ERayMode(Enum):
    DISTANCE = auto()
    SURFACE_FIELD = auto()
    LINE_OF_SIGHT = auto()
