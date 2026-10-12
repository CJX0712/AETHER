"""Supervised learning — reusing world-class closed-form & iterative methods.

  - LinearRegression / Ridge : normal equations (Gauss, 1821) + ridge (Tikhonov)
  - LogisticRegression       : sigmoid MLE solved by Adam (Kingma & Ba 2015)
  - KNeighborsClassifier     : brute-force k-NN (Cover & Hart 1967)
  - GaussianNB               : Naive Bayes w/ Gaussian features (MAP)

Author: 晨星
"""
import math
from ..core import linalg as L
from ..core.optim import adam


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LinearRegression:
    """Ordinary least squares via normal equations (no closed-form instability)."""

    def fit(self, X, y):
        n = len(X)
        Xb = [row + [1.0] for row in X]  # append bias column
        self.w = L.lstsq(Xb, y)
        self.coef_ = self.w[:-1]
        self.intercept_ = self.w[-1]
        return self

    def predict(self, X):
        return [sum(self.w[j] * x[j] for j in range(len(x))) + self.w[-1] for x in X]


class RidgeRegression:
    """Ridge regression: closed-form (XᵀX + λI)⁻¹ Xᵀy (Tikhonov regularization)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n = len(X)
        d = len(X[0])
        Xb = [row + [1.0] for row in X]
        Xt = L.transpose(Xb)
        XtX = L.matmul(Xt, Xb)
        for i in range(d + 1):
            XtX[i][i] += self.alpha
        Xty = [r[0] for r in L.matmul(Xt, [[yi] for yi in y])]
        self.w = L.solve(XtX, Xty)
        return self

    def predict(self, X):
        return [sum(self.w[j] * x[j] for j in range(len(x))) + self.w[-1] for x in X]


class LogisticRegression:
    """Binary logistic regression, MLE trained with Adam."""

    def __init__(self, lr=0.1, iters=2000):
        self.lr = lr
        self.iters = iters

    def fit(self, X, y):
        n = len(X)
        d = len(X[0])
        self.w = [0.0] * (d + 1)

        def loss_and_grad():
            g = [0.0] * (d + 1)
            ll = 0.0
            for i in range(n):
                z = self.w[-1] + sum(self.w[j] * X[i][j] for j in range(d))
                p = _sigmoid(z)
                ll += y[i] * math.log(p + 1e-12) + (1 - y[i]) * math.log(1 - p + 1e-12)
                err = p - y[i]
                for j in range(d):
                    g[j] += err * X[i][j]
                g[-1] += err
            return -ll / n, [gi / n for gi in g]

        def f(_):  # Adam only needs grad via closure; we re-evaluate
            return 0.0

        for _ in range(self.iters):
            _, g = loss_and_grad()
            # Adam step on w
            if not hasattr(self, "_m"):
                self._m = [0.0] * (d + 1)
                self._v = [0.0] * (d + 1)
                self._t = 0
            self._t += 1
            t = self._t
            b1, b2, eps = 0.9, 0.999, 1e-8
            self._m = [b1 * self._m[i] + (1 - b1) * g[i] for i in range(d + 1)]
            self._v = [b2 * self._v[i] + (1 - b2) * g[i] * g[i] for i in range(d + 1)]
            mh = [self._m[i] / (1 - b1 ** t) for i in range(d + 1)]
            vh = [self._v[i] / (1 - b2 ** t) for i in range(d + 1)]
            self.w = [self.w[i] - self.lr * mh[i] / (vh[i] ** 0.5 + eps) for i in range(d + 1)]
        return self

    def predict_proba(self, X):
        d = len(X[0])
        return [_sigmoid(self.w[-1] + sum(self.w[j] * x[j] for j in range(d))) for x in X]

    def predict(self, X):
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]


class KNeighborsClassifier:
    """Brute-force k-NN with Euclidean distance (Cover & Hart, 1967)."""

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def _dist(self, a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

    def predict(self, X):
        out = []
        for x in X:
            dists = sorted(
                (self._dist(x, self.X[i]), self.y[i]) for i in range(len(self.X))
            )
            votes = [d[1] for d in dists[: self.k]]
            out.append(max(set(votes), key=votes.count))
        return out


class GaussianNB:
    """Gaussian Naive Bayes (MAP classifier under conditional independence)."""

    def fit(self, X, y):
        self.classes = sorted(set(y))
        self.eps = 1e-9
        self.means, self.vars, self.priors = {}, {}, {}
        for c in self.classes:
            rows = [X[i] for i in range(len(X)) if y[i] == c]
            self.priors[c] = len(rows) / len(X)
            d = len(X[0])
            self.means[c] = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
            self.vars[c] = [
                sum((r[j] - self.means[c][j]) ** 2 for r in rows) / len(rows) + self.eps
                for j in range(d)
            ]
        return self

    def predict(self, X):
        import math as _m

        out = []
        for x in X:
            best, best_score = None, -1e18
            for c in self.classes:
                s = _m.log(self.priors[c])
                for j in range(len(x)):
                    m, v = self.means[c][j], self.vars[c][j]
                    s += -0.5 * _m.log(2 * _m.pi * v) - 0.5 * (x[j] - m) ** 2 / v
                if s > best_score:
                    best_score, best = s, c
            out.append(best)
        return out
