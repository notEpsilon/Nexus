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

    def __mul__(self, other: Union[Scalar, np.ndarray, Tensor]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        res = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad = np.array(self.grad) + other.data * res.grad
            other.grad = np.array(other.grad) + self.data * res.grad

        res._backward = _backward

        return res

    def __pow__(self, other: Scalar) -> Tensor:
        assert isinstance(
            other, (int, float)
        ), "only supports powers of (int, float) for now."
        res = Tensor(self.data**other, (self,))

        def _backward():
            self.grad = (
                np.array(self.grad) + (other * (self.data ** (other - 1))) * res.grad
            )

        res._backward = _backward

        return res

    def __matmul__(self, other: Union[np.ndarray, Tensor]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        res = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad = np.array(self.grad) + (res.grad @ other.data.T)
            other.grad = np.array(other.grad) + (self.data.T @ res.grad)

        res._backward = _backward

        return res

    def transpose(self):
        res = Tensor(np.transpose(self.data), (self,))

        def _backward():
            self.grad = np.array(self.grad) + np.transpose(res.grad)

        res._backward = _backward

        return res

    def mean(self):
        res = Tensor(np.mean(self.data), (self,))

        def _backward():
            self.grad = (
                np.array(self.grad)
                + (np.zeros_like(self.data) + (1 / np.size(self.data))) * res.grad
            )

        res._backward = _backward

        return res

    def backward(self) -> None:
        topo: List[Tensor] = []
        vis = set()

        def build_topo(node: Tensor) -> None:
            if node not in vis:
                vis.add(node)
                for ch in node._operands:
                    build_topo(ch)
                topo.append(node)

        build_topo(self)

        self.grad = 1
        for vertex in reversed(topo):
            vertex._backward()

    def linear(self, other: Union[np.ndarray, Tensor]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self @ other.transpose()

    def __neg__(self) -> Tensor:
        return self * -1

    def __sub__(self, other) -> Tensor:
        return self + (-other)

    def __radd__(self, other) -> Tensor:
        return self + other

    def __rmul__(self, other) -> Tensor:
        return self * other

    def __rsub__(self, other) -> Tensor:
        return (-self) + other

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad})"
