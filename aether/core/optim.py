"""First- and second-order optimizers — reusing canonical world-class methods.

  - Backtracking line search w/ Armijo condition (Nocedal & Wright, 2006)
  - Gradient descent (+ momentum)
  - Adam (Kingma & Ba, 2015) — adaptive moment estimation
  - Newton's method w/ exact Hessian solve (quadratic convergence)
  - Linear Conjugate Gradient (Hestenes & Stiefel, 1952)
  - BFGS quasi-Newton (Broyden-Fletcher-Goldfarb-Shanno) w/ two-loop recursion

All return (solution, loss_trace) so the test-suite can assert monotonic
descent / convergence invariants.

Author: 晨星
"""
from . import linalg as L


def line_search(f, grad, x, d, alpha0=1.0, rho=0.5, c=1e-4):
    """Backtracking Armijo line search along direction d from x."""
    fx = f(x)
    gx = grad(x)
    gd = sum(gx[i] * d[i] for i in range(len(x)))
    alpha = alpha0
    while alpha > 1e-14:
        xn = [x[i] + alpha * d[i] for i in range(len(x))]
        if f(xn) <= fx + c * alpha * gd:
            return alpha
        alpha *= rho
    return alpha


def gradient_descent(f, grad, x0, lr=0.1, iters=1000, momentum=0.0, tol=1e-9):
    x = list(x0)
    v = [0.0] * len(x)
    trace = []
    for _ in range(iters):
        g = grad(x)
        v = [momentum * v[i] - lr * g[i] for i in range(len(x))]
        x = [x[i] + v[i] for i in range(len(x))]
        fx = f(x)
        trace.append(fx)
        if max(abs(g[i]) for i in range(len(x))) < tol:
            break
    return x, trace


def adam(f, grad, x0, iters=2000, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, tol=1e-9):
    """Adam optimizer (Kingma & Ba, 2015)."""
    x = list(x0)
    m = [0.0] * len(x)
    v = [0.0] * len(x)
    trace = []
    for t in range(1, iters + 1):
        g = grad(x)
        m = [b1 * m[i] + (1 - b1) * g[i] for i in range(len(x))]
        v = [b2 * v[i] + (1 - b2) * g[i] * g[i] for i in range(len(x))]
        mh = [m[i] / (1 - b1 ** t) for i in range(len(x))]
        vh = [v[i] / (1 - b2 ** t) for i in range(len(x))]
        x = [x[i] - lr * mh[i] / (v[i] ** 0.5 + eps) for i in range(len(x))]
        fx = f(x)
        trace.append(fx)
        if max(abs(g[i]) for i in range(len(x))) < tol:
            break
    return x, trace


def newton(f, grad, hess, x0, iters=100, tol=1e-12):
    """Newton's method using exact Hessian solve (quadratic convergence)."""
    x = list(x0)
    trace = []
    for _ in range(iters):
        g = grad(x)
        H = hess(x)
        try:
            d = L.solve(H, [-gi for gi in g])
        except ValueError:
            d = [-gi for gi in g]
        alpha = line_search(f, grad, x, d)
        x = [x[i] + alpha * d[i] for i in range(len(x))]
        fx = f(x)
        trace.append(fx)
        if max(abs(g[i]) for i in range(len(x))) < tol:
            break
    return x, trace


def conjugate_gradient(A, b, iters=None, tol=1e-12):
    """Linear Conjugate Gradient for SPD system A x = b.

    Converges to the exact solution in ≤ n iterations (n = len(b)).
    """
    n = len(b)
    iters = iters or n
    x = [0.0] * n
    r = [b[i] - L.matvec(A, x)[i] for i in range(n)]
    p = list(r)
    rs_old = sum(r[i] * r[i] for i in range(n))
    trace = []
    for _ in range(iters):
        Ap = L.matvec(A, p)
        alpha = rs_old / sum(p[i] * Ap[i] for i in range(n))
        x = [x[i] + alpha * p[i] for i in range(n)]
        r = [r[i] - alpha * Ap[i] for i in range(n)]
        rs_new = sum(r[i] * r[i] for i in range(n))
        trace.append(rs_new)
        if rs_new < tol:
            break
        beta = rs_new / rs_old
        p = [r[i] + beta * p[i] for i in range(n)]
        rs_old = rs_new
    return x, trace


def bfgs(f, grad, x0, iters=1000, lr=1.0, tol=1e-9):
    """BFGS quasi-Newton with two-loop recursion on the inverse Hessian."""
    n = len(x0)
    x = list(x0)
    I = L.identity(n)
    H = [row[:] for row in I]  # inverse Hessian approximation
    g = grad(x)
    trace = []
    for _ in range(iters):
        d = [-sum(H[i][j] * g[j] for j in range(n)) for i in range(n)]
        alpha = line_search(f, grad, x, d)
        s = [alpha * d[i] for i in range(n)]
        x_new = [x[i] + s[i] for i in range(n)]
        g_new = grad(x_new)
        y = [g_new[i] - g[i] for i in range(n)]
        sTy = sum(s[i] * y[i] for i in range(n))
        if abs(sTy) < 1e-14:
            break
        # BFGS rank-2 update of H
        Hy = [sum(H[i][j] * y[j] for j in range(n)) for i in range(n)]
        yTHy = sum(y[i] * Hy[i] for i in range(n))
        rho = 1.0 / sTy
        H = [
            [
                H[i][j]
                + rho * s[i] * s[j]
                - (Hy[i] * y[j] + y[i] * Hy[j]) / yTHy
                + rho * rho * yTHy * s[i] * s[j]
                for j in range(n)
            ]
            for i in range(n)
        ]
        x, g = x_new, g_new
        fx = f(x)
        trace.append(fx)
        if max(abs(g[i]) for i in range(n)) < tol:
            break
    return x, trace
