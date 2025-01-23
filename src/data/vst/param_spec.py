import random
from typing import Any, List, Optional, Tuple

import numpy as np


class Parameter:
    name: str

    def __init__(self, name: str):
        self.name = name

    def sample(self) -> float:
        raise NotImplementedError


class DiscreteParameter(Parameter):
    def __init__(
        self,
        name: str,
        values: List[Any],
        raw_values: Optional[List[Any]] = None,
        weights: Optional[List[float]] = None,
    ):
        super().__init__(name)

        if raw_values is not None:
            assert len(values) == len(
                raw_values
            ), "values and raw_values must have the same length"

        else:
            n = len(values)
            raw_values = [i / (n - 1) for i in range(n)]

        if weights is not None:
            assert len(values) == len(
                weights
            ), "values and weights must have the same length"

        else:
            weights = [1.0] * len(values)

        self.values = values
        self.raw_values = raw_values
        self.weights = weights

    def sample(self) -> float:
        p = np.array(self.weights)
        p /= p.sum()
        return np.random.choice(self.raw_values, p=p)

    def __repr__(self):
        return f'DiscreteParameter(name="{self.name}", values={self.values}, raw_values={self.raw_values})'


class ContinuousParameter(Parameter):
    def __init__(
        self,
        name: str,
        min: float = 0.0,
        max: float = 1.0,
        constant_val_p: float = 0.0,
        constant_val: float = 0.0,
    ):
        super().__init__(name)

        assert max > min, "max must be greater than min"
        assert min >= 0.0, "min must be greater than or equal to 0.0"
        assert max <= 1.0, "max must be less than or equal to 1.0"

        self.min = min
        self.max = max

        self.constant_val_p = constant_val_p
        self.constant_val = constant_val

    def sample(self) -> float:
        if self.constant_val_p > 0.0 and random.random() < self.constant_val_p:
            return self.constant_val

        return random.uniform(self.min, self.max)

    def __repr__(self):
        return (
            f'ContinuousParameter(name="{self.name}", min={self.min}, max={self.max})'
        )


class ParamSpec(list):
    def sample(self) -> dict[str, float]:
        return {p.name: p.sample() for p in self}

    def to_numpy(
        self,
        param_dict: dict[str, float],
        note: Optional[int] = None,
        min_pitch: int = 36,
        max_pitch: int = 84,
    ) -> np.ndarray:
        if note is None:
            note = 0.0
        else:
            note = (note - min_pitch) / (max_pitch - min_pitch)

        params = [param_dict[p.name] for p in self]
        params += [note]

        return np.array(params, dtype=np.float32)

    def from_numpy(
        self, params: np.ndarray, min_pitch: int = 36, max_pitch: int = 84
    ) -> Tuple[dict[str, float], int]:
        note = params[-1] * (max_pitch - min_pitch) + min_pitch
        return {p.name: params[i] for i, p in enumerate(self)}, note
