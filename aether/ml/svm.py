"""Support Vector Machine via Platt's Sequential Minimal Optimization (SMO).

Binary classification, linear kernel. Reuses Platt (1998) SMO: optimize two
Lagrange multipliers at a time, maintaining KKT optimality. On convergence the
support vectors satisfy  yᵢ f(xᵢ) = 1  (for 0 < αᵢ < C). The dual objective
is maximized.

Author: 晨星
"""
import math


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_passes=20):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes

    def fit(self, X, y):
        self.X = X
        self.y = [1 if yi > 0 else -1 for yi in y]
        n = len(X)
        self.alphas = [0.0] * n
        self.b = 0.0
        # linear kernel matrix (PSD by construction)
        self.K = [[_dot(X[i], X[j]) for j in range(n)] for i in range(n)]
        E = [-self.y[i] for i in range(n)]  # f starts at 0 ⇒ Eᵢ = 0 − yᵢ
        passes = 0
        rng_state = 12345
        while passes < self.max_passes:
            num_changed = 0
            for i2 in range(n):
                y2, a2, E2 = self.y[i2], self.alphas[i2], E[i2]
                r2 = E2 * y2
                if (r2 < -self.tol and a2 < self.C) or (r2 > self.tol and a2 > 0):
                    # heuristic: pick i1 with largest |E1 − E2| among free alphas
                    i1, best = -1, -1.0
                    for i in range(n):
                        if 0 < self.alphas[i] < self.C:
                            diff = abs(E[i] - E2)
                            if diff > best:
                                best, i1 = diff, i
                    if i1 >= 0 and self._take_step(i1, i2, E):
                        num_changed += 1
                        self._refresh_E(E)
                        continue
                    # fallback: try ANY i1 ≠ i2 (so the first step can seed)
                    for i in range(n):
                        if i == i2:
                            continue
                        if self._take_step(i, i2, E):
                            num_changed += 1
                            self._refresh_E(E)
                            break
            if num_changed == 0:
                passes += 1
            else:
                passes = 0
            # refresh error cache (alphas changed)
            for i in range(n):
                E[i] = self._f(X[i]) - self.y[i]
        self.sv = [i for i in range(n) if self.alphas[i] > 1e-5]
        return self

    def _f(self, x):
        s = self.b
        for i in range(len(self.X)):
            if self.alphas[i] > 1e-9:
                s += self.alphas[i] * self.y[i] * _dot(self.X[i], x)
        return s

    def _refresh_E(self, E):
        for i in range(len(self.X)):
            E[i] = self._f(self.X[i]) - self.y[i]

    def _take_step(self, i1, i2, E):
        if i1 == i2:
            return False
        a1, y1 = self.alphas[i1], self.y[i1]
        a2, y2 = self.alphas[i2], self.y[i2]
        # recompute errors from the CURRENT model (no stale cache → stable b)
        E1 = self._f(self.X[i1]) - y1
        E2 = self._f(self.X[i2]) - y2
        s = y1 * y2
        if s < 0:
            L = max(0.0, a2 - a1)
            H = min(self.C, self.C + a2 - a1)
        else:
            L = max(0.0, a2 + a1 - self.C)
            H = min(self.C, a2 + a1)
        if L >= H:
            return False
        k11, k12, k22 = self.K[i1][i1], self.K[i1][i2], self.K[i2][i2]
        eta = 2 * k12 - k11 - k22
        if eta < 0:
            a2_new = a2 - y2 * (E1 - E2) / eta
            if a2_new < L:
                a2_new = L
            elif a2_new > H:
                a2_new = H
        else:
            # objective is linear on [L, H]; pick endpoint minimizing it
            c1 = eta / 2.0
            c2 = y2 * (E1 - E2) - eta * a2
            Lobj, Hobj = c1 * L * L + c2 * L, c1 * H * H + c2 * H
            a2_new = L if Lobj > Hobj else (H if Lobj < Hobj else a2)
        if abs(a2_new - a2) < 1e-5 * (a2_new + a2 + 1e-5):
            return False
        a1_new = a1 + s * (a2 - a2_new)
        b1 = self.b - E1 - y1 * (a1_new - a1) * k11 - y2 * (a2_new - a2) * k12
        b2 = self.b - E2 - y1 * (a1_new - a1) * k12 - y2 * (a2_new - a2) * k22
        if 0 < a1_new < self.C:
            self.b = b1
        elif 0 < a2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0
        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new
        return True

    def predict(self, X):
        return [1 if self._f(x) >= 0 else 0 for x in X]
