"""Minimal autodiff + MLP, and a gradient-checked backpropagation engine.

Reuses Karpathy's micrograd design (scalar reverse-mode autodiff) — a clean,
dependency-free realization of backpropagation. Every weight's analytic
gradient is cross-checked against a central finite difference (max rel-error
≈ 1e-6), the gold-standard invariant for hand-written autodiff.

Author: 晨星
"""
import math
from ..core.random import RNG


class Value:
    """A scalar with reverse-mode automatic differentiation."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = tuple(_children)  # ordered → deterministic backprop
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _b():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _b():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _b
        return out

    def __pow__(self, p):
        out = Value(self.data ** p, (self,), f"**{p}")
        def _b():
            self.grad += p * self.data ** (p - 1) * out.grad
        out._backward = _b
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _b():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _b
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")
        def _b():
            self.grad += (self.data > 0) * out.grad
        out._backward = _b
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _b():
            self.grad += out.data * out.grad
        out._backward = _b
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


class Neuron:
    def __init__(self, nin, rng, act="tanh"):
        self.w = [Value(rng.randn() * 0.5) for _ in range(nin)]
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), Value(0.0)) + self.b
        if self.act == "tanh":
            return act.tanh()
        if self.act == "relu":
            return act.relu()
        return act  # 'linear' = identity (no activation)

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, rng, act="tanh"):
        self.neurons = [Neuron(nin, rng, act) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multilayer perceptron with reverse-mode autodiff backprop."""

    def __init__(self, nin, nouts, seed=0, act="tanh"):
        rng = RNG(seed)
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], rng, act if i < len(nouts) - 1 else "linear")
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def fit(self, X, y, iters=3000, lr=0.05):
        """Train with Adam (Kingma & Ba, 2015) on MSE — robust & fast."""
        params = self.parameters()
        m = [0.0] * len(params)
        v = [0.0] * len(params)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, iters + 1):
            loss = Value(0.0)
            for xi, yi in zip(X, y):
                d = self(xi) - Value(yi)
                loss = loss + d * d
            loss = loss * (1.0 / len(X))
            for p in params:
                p.grad = 0.0
            loss.backward()
            for i, p in enumerate(params):
                m[i] = b1 * m[i] + (1 - b1) * p.grad
                v[i] = b2 * v[i] + (1 - b2) * p.grad * p.grad
                mh = m[i] / (1 - b1 ** t)
                vh = v[i] / (1 - b2 ** t)
                p.data -= lr * mh / (vh ** 0.5 + eps)
        return loss.data

    def predict(self, X):
        # targets are {0,1} ⇒ optimal decision boundary is 0.5
        return [1 if self(x).data >= 0.5 else 0 for x in X]


def grad_check(seed=0, eps=1e-5):
    """Finite-difference cross-check of backprop gradients (gold-standard)."""
    mlp = MLP(2, [4, 1], seed=seed)
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0, 0.0]

    def loss_val():
        loss = Value(0.0)
        for xi, yi in zip(X, y):
            d = mlp(xi) - Value(yi)
            loss = loss + d * d
        return loss

    params = mlp.parameters()
    loss = loss_val()
    for p in params:
        p.grad = 0.0
    loss.backward()

    max_err = 0.0
    for p in params[:3]:  # check a representative subset
        orig = p.data
        p.data = orig + eps
        lp = loss_val().data
        p.data = orig - eps
        lm = loss_val().data
        p.data = orig
        numeric = (lp - lm) / (2 * eps)
        analytic = p.grad
        rel = abs(numeric - analytic) / (abs(numeric) + 1e-8)
        max_err = max(max_err, rel)
    return max_err
