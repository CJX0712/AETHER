"""Principal Component Analysis via symmetric eigendecomposition.

Covariance matrix → Jacobi eigen-decomposition (orthogonal eigenvectors) →
sorted principal axes. Invariants: components are orthonormal, and explained
variance equals the eigenvalues of the covariance matrix.

Author: 晨星
"""
from ..core import linalg as L


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        n = len(X)
        d = len(X[0])
        self.mean = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        Xc = [[X[i][j] - self.mean[j] for j in range(d)] for i in range(n)]
        Xt = L.transpose(Xc)
        cov = L.matmul(Xt, Xc)
        cov = [[cov[a][b] / n for b in range(d)] for a in range(d)]
        eigvals, eigvecs = L.jacobi_eigen(cov)  # eigvecs[c] = eigenvector c
        order = sorted(range(d), key=lambda i: -eigvals[i])
        self.explained_variance_ = [eigvals[i] for i in order]
        # components as rows, ordered by descending variance
        self.components_ = [eigvecs[i][:] for i in order]
        total = sum(self.explained_variance_) or 1.0
        self.explained_variance_ratio_ = [v / total for v in self.explained_variance_]
        if self.n_components is not None:
            self.components_ = self.components_[: self.n_components]
            self.explained_variance_ = self.explained_variance_[: self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[: self.n_components]
        return self

    def transform(self, X):
        comps = self.components_
        out = []
        for x in X:
            xc = [x[j] - self.mean[j] for j in range(len(x))]
            out.append([sum(xc[k] * comps[c][k] for k in range(len(xc))) for c in range(len(comps))])
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)
