"""
    Simple tensor implementation using numpy `ndarray`
"""
from __future__ import annotations

import numpy as np

from typing import Tuple, Union, List, Any


Scalar = Union[float, int]


class Tensor:
    def __init__(
        self, data: Union[Scalar, np.ndarray, List[Any]], _operands: Tuple[Tensor] = ()
    ) -> None:
        self.data = np.array(data)
        self.grad: Union[Scalar, np.ndarray] = 0
        self._operands = _operands
        self._backward = lambda: None

    def __add__(self, other: Union[Scalar, np.ndarray, Tensor]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        res = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad = np.array(self.grad) + res.grad
            other.grad = np.array(other.grad) + res.grad

        res._backward = _backward

        return res
