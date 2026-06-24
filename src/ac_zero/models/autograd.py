from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _unbroadcast(grad: Array, shape: tuple[int, ...]) -> Array:
    """Sum a gradient back to ``shape`` to reverse NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Node:
    """A node in a small reverse-mode autodiff graph over 2-D float arrays.

    Every operation returns a new ``Node`` and registers a closure that
    propagates the upstream gradient to its parents. Keeping every value 2-D
    makes the matmul and broadcasting rules uniform, which is enough for the
    compact CPU policy/value trunks in this package.
    """

    __slots__ = ("_backward", "_parents", "data", "grad", "requires_grad")

    def __init__(
        self,
        data: Array | float,
        parents: tuple[Node, ...] = (),
        requires_grad: bool = False,
    ) -> None:
        self.data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        self.grad = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._parents = parents
        self.requires_grad: bool = bool(requires_grad) or any(p.requires_grad for p in parents)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __add__(self, other: Node | float) -> Node:
        rhs = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + rhs.data, (self, rhs))

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.data.shape)
            rhs.grad += _unbroadcast(out.grad, rhs.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other: Node | float) -> Node:
        rhs = other if isinstance(other, Node) else Node(other)
        return self + (-rhs)

    def __neg__(self) -> Node:
        out = Node(-self.data, (self,))

        def _backward() -> None:
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Node | float) -> Node:
        rhs = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * rhs.data, (self, rhs))

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad * rhs.data, self.data.shape)
            rhs.grad += _unbroadcast(out.grad * self.data, rhs.data.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other: Node | float) -> Node:
        return (-self) + other

    __radd__ = __add__
    __rmul__ = __mul__

    def matmul(self, other: Node) -> Node:
        out = Node(self.data @ other.data, (self, other))

        def _backward() -> None:
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: Node) -> Node:
        return self.matmul(other)

    def transpose(self) -> Node:
        out = Node(self.data.T, (self,))

        def _backward() -> None:
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def relu(self) -> Node:
        out = Node(np.maximum(self.data, 0.0), (self,))

        def _backward() -> None:
            self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Node:
        value = np.tanh(self.data)
        out = Node(value, (self,))

        def _backward() -> None:
            self.grad += (1.0 - value**2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Node:
        value = 1.0 / (1.0 + np.exp(-self.data))
        out = Node(value, (self,))

        def _backward() -> None:
            self.grad += value * (1.0 - value) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Node:
        if axis is None:
            reduced = np.asarray(self.data.sum(), dtype=np.float64)
        else:
            summed = self.data.sum(axis=axis)
            reduced = np.expand_dims(summed, axis) if keepdims else summed
        out = Node(reduced, (self,))

        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis: int | None = None, keepdims: bool = False) -> Node:
        count = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(count))

    def softmax_rows(self) -> Node:
        """Row-wise softmax with the standard analytic Jacobian backward."""
        shifted = self.data - self.data.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        out = Node(probs, (self,))

        def _backward() -> None:
            dot = (out.grad * probs).sum(axis=1, keepdims=True)
            self.grad += probs * (out.grad - dot)

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Run reverse-mode accumulation from this (scalar) node."""
        topo: list[Node] = []
        seen: set[int] = set()

        def build(node: Node) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


def concat_rows(nodes: Iterable[Node]) -> Node:
    """Stack single-row nodes into one ``(n, width)`` node."""
    items = list(nodes)
    out = Node(np.concatenate([n.data for n in items], axis=0), tuple(items))

    def _backward() -> None:
        for index, item in enumerate(items):
            item.grad += out.grad[index : index + 1]

    out._backward = _backward
    return out


def concat_cols(nodes: Iterable[Node]) -> Node:
    """Concatenate single-row nodes along their columns into one wide row."""
    items = list(nodes)
    widths = [n.data.shape[1] for n in items]
    out = Node(np.concatenate([n.data for n in items], axis=1), tuple(items))

    def _backward() -> None:
        start = 0
        for item, width in zip(items, widths, strict=True):
            item.grad += out.grad[:, start : start + width]
            start += width

    out._backward = _backward
    return out


def embedding_lookup(table: Node, indices: NDArray[np.int64]) -> Node:
    """Gather rows ``table[indices]`` with a scatter-add backward pass."""
    rows = np.asarray(indices, dtype=np.int64)
    out = Node(table.data[rows], (table,))

    def _backward() -> None:
        np.add.at(table.grad, rows, out.grad)

    out._backward = _backward
    return out
