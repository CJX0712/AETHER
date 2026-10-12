"""Unsupervised learning — k-means++ and Gaussian Mixture Models (EM).

  - KMeans   : k-means++ seeding (Arthur & Vassilvitskii, 2007) + Lloyd's
  - GMM      : Expectation-Maximization (Dempster, Laird & Rubin, 1977)
               with the monotone log-likelihood (ELBO) guarantee.

Densities use the Cholesky factorization for numerical stability (no raw
matrix inverse):  ‖Σ⁻¹ᐟ²(x-μ)‖²  and  log|Σ| = 2 Σ log Lᵢᵢ.

Author: 晨星
"""
import math
from ..core import linalg as L
from ..core.random import RNG


def _inertia(X, centers):
    s = 0.0
    for x in X:
        best = min(sum((x[j] - c[j]) ** 2 for j in range(len(x))) for c in centers)
        s += best
    return s


class KMeans:
    """k-means++ with Lloyd's algorithm. Inertia never increases."""

    def __init__(self, k=3, seed=0, max_iter=300):
        self.k = k
        self.seed = seed
        self.max_iter = max_iter

    def fit(self, X):
        rng = RNG(self.seed)
        n, d = len(X), len(X[0])
        centers = [list(X[rng.randint(0, n)])]
        # k-means++ seeding
        for _ in range(1, self.k):
            dists = [
                min(sum((X[i][j] - c[j]) ** 2 for j in range(d)) for c in centers)
                for i in range(n)
            ]
            total = sum(dists)
            r = rng.rand() * total
            acc = 0.0
            for i in range(n):
                acc += dists[i]
                if acc >= r:
                    centers.append(list(X[i]))
                    break
        for _ in range(self.max_iter):
            labels = [
                min(range(self.k), key=lambda c: sum((X[i][j] - centers[c][j]) ** 2 for j in range(d)))
                for i in range(n)
            ]
            new = []
            for c in range(self.k):
                pts = [X[i] for i in range(n) if labels[i] == c]
                if pts:
                    new.append([sum(p[j] for p in pts) / len(pts) for j in range(d)])
                else:
                    new.append(list(centers[c]))
            if all(
                sum((new[c][j] - centers[c][j]) ** 2 for j in range(d)) < 1e-9
                for c in range(self.k)
            ):
                centers = new
                break
            centers = new
        self.centers = centers
        self.labels_ = [
            min(range(self.k), key=lambda c: sum((X[i][j] - centers[c][j]) ** 2 for j in range(d)))
            for i in range(n)
        ]
        return self

    def predict(self, X):
        d = len(X[0])
        return [
            min(range(self.k), key=lambda c: sum((x[j] - self.centers[c][j]) ** 2 for j in range(d)))
            for x in X
        ]


class GMM:
    """Gaussian Mixture Model fit by Expectation-Maximization."""

    def __init__(self, k=2, seed=0, max_iter=200, reg=1e-6):
        self.k = k
        self.seed = seed
        self.max_iter = max_iter
        self.reg = reg

    @staticmethod
    def _chol_logpdf(x, mu, L, d):
        # forward-substitute L y = (x - mu)
        diff = [x[j] - mu[j] for j in range(d)]
        y = [0.0] * d
        for i in range(d):
            s = diff[i]
            for j in range(i):
                s -= L[i][j] * y[j]
            y[i] = s / L[i][i]
        quad = sum(v * v for v in y)
        logdet = 2.0 * sum(math.log(L[i][i]) for i in range(d))
        return -0.5 * (d * math.log(2 * math.pi) + logdet + quad)

    def fit(self, X):
        rng = RNG(self.seed)
        n, d = len(X), len(X[0])
        # init means via random points, equal weights, identity cov
        idx = rng.shuffle(list(range(n)))[: self.k]
        self.mu = [list(X[i]) for i in idx]
        self.pi = [1.0 / self.k] * self.k
        self.L = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        self.Ls = [[[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)] for _ in range(self.k)]
        self.ll_trace = []
        for _ in range(self.max_iter):
            # E-step
            log_resp = [[0.0] * self.k for _ in range(n)]
            ll = 0.0
            for i in range(n):
                scores = [
                    math.log(self.pi[c] + 1e-300) + self._chol_logpdf(X[i], self.mu[c], self.Ls[c], d)
                    for c in range(self.k)
                ]
                m = max(scores)
                lse = m + math.log(sum(math.exp(s - m) for s in scores))
                ll += lse
                for c in range(self.k):
                    log_resp[i][c] = math.exp(scores[c] - lse)
            self.ll_trace.append(ll)
            # M-step
            Nk = [sum(log_resp[i][c] for i in range(n)) for c in range(self.k)]
            for c in range(self.k):
                self.mu[c] = [
                    sum(log_resp[i][c] * X[i][j] for i in range(n)) / (Nk[c] + 1e-12)
                    for j in range(d)
                ]
                cov = [[0.0] * d for _ in range(d)]
                for i in range(n):
                    for a in range(d):
                        for b in range(d):
                            cov[a][b] += log_resp[i][c] * (X[i][a] - self.mu[c][a]) * (X[i][b] - self.mu[c][b])
                cov = [[cov[a][b] / (Nk[c] + 1e-12) + (self.reg if a == b else 0.0) for b in range(d)] for a in range(d)]
                self.Ls[c] = L.cholesky(cov)
            self.pi = [Nk[c] / n for c in range(self.k)]
            # convergence on log-likelihood
            if len(self.ll_trace) >= 2 and abs(self.ll_trace[-1] - self.ll_trace[-2]) < 1e-9:
                break
        return self

    def predict(self, X):
        d = len(X[0])
        out = []
        for x in X:
            best, bs = 0, -1e18
            for c in range(self.k):
                s = math.log(self.pi[c] + 1e-300) + self._chol_logpdf(x, self.mu[c], self.Ls[c], d)
                if s > bs:
                    bs, best = s, c
            out.append(best)
        return out
