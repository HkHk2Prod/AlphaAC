import numpy as np

from ac_zero.models.autograd import Node, concat_cols, embedding_lookup


def _numeric_grad(func, node, eps=1e-6):
    grad = np.zeros_like(node.data)
    it = np.nditer(node.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = node.data[idx]
        node.data[idx] = original + eps
        plus = float(func().data.sum())
        node.data[idx] = original - eps
        minus = float(func().data.sum())
        node.data[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def test_reverse_mode_matches_finite_differences_across_all_ops() -> None:
    rng = np.random.default_rng(0)
    weight = Node(rng.normal(size=(3, 4)), requires_grad=True)
    bias = Node(rng.normal(size=(1, 4)), requires_grad=True)
    table = Node(rng.normal(size=(5, 3)), requires_grad=True)
    indices = np.array([1, 3, 3, 0], dtype=np.int64)

    def forward() -> Node:
        embedded = embedding_lookup(table, indices)
        hidden = (embedded @ weight + bias).relu()
        attention = (hidden @ hidden.transpose()).softmax_rows()
        pooled = (attention @ hidden).mean(axis=0, keepdims=True)
        widened = concat_cols([pooled, pooled.tanh()])
        return (widened * widened).sum()

    loss = forward()
    loss.backward()
    for node in (weight, bias, table):
        numeric = _numeric_grad(forward, node)
        assert np.max(np.abs(numeric - node.grad)) < 1e-6


def test_sigmoid_and_subtraction_backward() -> None:
    x = Node([[0.5, -1.0]], requires_grad=True)
    loss = (1.0 - x.sigmoid()).sum()
    loss.backward()
    sig = 1.0 / (1.0 + np.exp(-x.data))
    assert np.allclose(x.grad, -(sig * (1.0 - sig)))


def test_softmax_rows_normalizes() -> None:
    logits = Node([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = logits.softmax_rows()
    assert np.allclose(probs.data.sum(axis=1), 1.0)
    assert np.allclose(probs.data[1], np.full(3, 1 / 3))
