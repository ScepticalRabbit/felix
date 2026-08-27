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
