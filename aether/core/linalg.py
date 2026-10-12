"""Pure-Python numerical linear algebra on top of world-class algorithms.

All matrices are row-major lists of lists. No BLAS dependency — the engine
reuses canonical, numerically-stable methods:

  - Gaussian elimination w/ partial pivoting  (solve linear systems)
  - Cholesky decomposition                   (A = L Lᵀ, SPD matrices)
  - Jacobi eigenvalue algorithm (Jacobi 1846, revived by Goldstine/Householder)
    → symmetric eigen-decomposition with *orthogonal* eigenvectors
  - SVD via covariance eigendecomposition + orthonormalization

Every routine exposes an analytic invariant that the test-suite cross-checks
(e.g. A V ≈ V Λ,  UᵀU = I,  L Lᵀ = A).

Author: 晨星
"""
import math
from .random import RNG


# ---------- elementary ops ----------
def transpose(A):
    return [list(col) for col in zip(*A)]


def matmul(A, B):
    n, m = len(A), len(B[0])
    k = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for p in range(k):
            a = Ai[p]
            if a == 0.0:
                continue
            Bp = B[p]
            for j in range(m):
                Ci[j] += a * Bp[j]
    return C


def add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def scale(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def dot(u, v):
    return sum(u[i] * v[i] for i in range(len(u)))


def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


def matvec(A, x):
    return [dot(A[i], x) for i in range(len(A))]


# ---------- linear systems ----------
def solve(A, b, eps=1e-12):
    """Solve A x = b via Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        # partial pivot
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < eps:
            raise ValueError("singular matrix")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pv
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= f * M[col][j]
    return [M[i][n] for i in range(n)]


def cholesky(A, eps=1e-12):
    """Lower-triangular L with L Lᵀ = A (A symmetric positive-definite)."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= eps:
                    raise ValueError("matrix not positive-definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


# ---------- symmetric eigen-decomposition (Jacobi) ----------
def _jacobi_rotate(S, p, q, n):
    """One Jacobi step: similarity transform S ← Jᵀ S J that zeros S[p][q].

    Standard symmetric update (Golub & Van Loan). Returns the updated S, the
    rotation J (so V ← V J accumulates eigenvectors), and the cosine/sine (c, s).
    """
    if abs(S[p][q]) < 1e-300:
        return S, identity(n), 1.0, 0.0
    app, aqq, apq = S[p][p], S[q][q], S[p][q]
    phi = (aqq - app) / (2.0 * apq)
    t = math.copysign(1.0, phi) / (abs(phi) + math.sqrt(phi * phi + 1.0))
    c = 1.0 / math.sqrt(t * t + 1.0)
    s = t * c
    # 2x2 block on (p, q): Jᵀ A J with J = [[c, s], [-s, c]]
    S[p][p] = c * c * app - 2 * c * s * apq + s * s * aqq
    S[q][q] = s * s * app + 2 * c * s * apq + c * c * aqq
    S[p][q] = 0.0
    S[q][p] = 0.0
    # remaining rows/columns p, q (symmetry exploited)
    for k in range(n):
        if k == p or k == q:
            continue
        spk, sqk = S[p][k], S[q][k]
        S[p][k] = c * spk - s * sqk
        S[q][k] = s * spk + c * sqk
        S[k][p] = S[p][k]
        S[k][q] = S[q][k]
    J = identity(n)
    J[p][p] = c; J[q][q] = c
    J[p][q] = s; J[q][p] = -s
    return S, J, c, s


def jacobi_eigen(A, max_sweeps=200, tol=1e-12):
    """Eigen-decomposition of a symmetric matrix.

    Returns (eigenvalues, eigenvectors) where eigenvectors are COLUMN vectors,
    i.e. V[:, i] is the eigenvector for eigenvalues[i], and V is orthogonal.
    """
    n = len(A)
    S = [row[:] for row in A]
    V = identity(n)
    for _ in range(max_sweeps):
        off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off += S[p][q] * S[p][q]
        if off < tol:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(S[p][q]) > 1e-14:
                    S, _J, c, s = _jacobi_rotate(S, p, q, n)
                    # accumulate eigenvectors: V = V J
                    for r in range(n):
                        vr = V[r]
                        a = vr[p]; b = vr[q]
                        vr[p] = c * a - s * b
                        vr[q] = s * a + c * b
    eig = [S[i][i] for i in range(n)]
    return eig, transpose(V)  # V columns as rows-of-rows => transpose to columns


# ---------- SVD ----------
def _gram_schmidt(U):
    """Orthonormalize columns of U (modified Gram-Schmidt)."""
    m, k = len(U), len(U[0])
    Q = [[0.0] * k for _ in range(m)]
    for j in range(k):
        v = [U[i][j] for i in range(m)]
        for p in range(j):
            q = [Q[i][p] for i in range(m)]
            proj = sum(v[i] * q[i] for i in range(m))
            for i in range(m):
                v[i] -= proj * q[i]
        norm = math.sqrt(sum(x * x for x in v))
        if norm < 1e-12:
            continue
        for i in range(m):
            Q[i][j] = v[i] / norm
    return Q


def svd(A, max_sweeps=200, tol=1e-12):
    """Thin SVD: A (m×n) = U (m×k) · diag(S) · Vᵀ (n×k), k = min(m, n).

    Method: eigendecompose B = AᵀA (symmetric → Jacobi), V = eigenvectors,
    singular values σ = √λ. U = A V Σ⁻¹ then orthonormalized.
    """
    m = len(A)
    n = len(A[0])
    k = min(m, n)
    At = transpose(A)
    B = matmul(At, A)  # n×n symmetric
    eig, Vcols = jacobi_eigen(B, max_sweeps, tol)  # Vcols[i] = eigenvector i
    # pair eigenvalues with eigenvectors, sort descending by magnitude
    pairs = sorted(
        ((eig[i], Vcols[i]) for i in range(n)),
        key=lambda p: -p[0],
    )
    S = [math.sqrt(max(p[0], 1e-12)) for p in pairs[:k]]
    V = transpose([p[1] for p in pairs[:k]])  # n×k
    # U = A V Σ⁻¹
    AV = matmul(A, V)  # m×k
    U = [[AV[i][j] / S[j] for j in range(k)] for i in range(m)]
    U = _gram_schmidt(U)
    return U, S, V


# ---------- least squares (normal equations, ridge-stabilized) ----------
def lstsq(X, y, ridge=1e-9):
    """Solve min_w ||X w - y||²  →  w (normal equations + tiny ridge)."""
    Xt = transpose(X)
    XtX = matmul(Xt, X)
    n = len(XtX)
    for i in range(n):
        XtX[i][i] += ridge
    Xty = matmul(Xt, [[yi] for yi in y])
    Xty = [r[0] for r in Xty]
    return solve(XtX, Xty)
