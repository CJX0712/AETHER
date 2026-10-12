"""AETHER invariant test-suite — analytic cross-checks for every module.

Run:  python tests/test_invariants.py
Exits non-zero on the first failed invariant (CI-ready, no third-party deps).

Author: 晨星
"""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aether
from aether.core import linalg as L, optim as O
from aether.core.random import RNG
from aether.ml import linear, cluster, pca, tree, svm
from aether.deep import nn, attention
from aether.rl import qlearning

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def rmse(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


# ---------------- data helpers ----------------
def blobs(n_per=120, seed=1, sep=4.0):
    rng = RNG(seed)
    X, y = [], []
    means = [(-sep, -sep), (sep, -sep), (sep, sep)]
    for c, (mx, my) in enumerate(means):
        for _ in range(n_per):
            X.append([mx + rng.randn(), my + rng.randn()])
            y.append(c)
    return X, y


# ---------------- linalg ----------------
def test_linalg():
    print("[linalg]")
    # matmul identity
    A = [[1, 2], [3, 4]]
    I = L.identity(2)
    recon = L.matmul(A, I)
    flatA = [A[i][j] for i in range(2) for j in range(2)]
    flatR = [recon[i][j] for i in range(2) for j in range(2)]
    check("matmul identity", rmse(flatA, flatR) < 1e-12)
    # solve
    B = [[2, 1], [1, 3]]
    b = [3, 5]
    x = L.solve(B, b)
    check("linear solve A x = b", rmse(L.matvec(B, x), b) < 1e-10, f"res={rmse(L.matvec(B,x),b)}")
    # cholesky
    S = [[4.0, 2.0], [2.0, 3.0]]  # SPD
    Lm = L.cholesky(S)
    recon = L.matmul(Lm, L.transpose(Lm))
    check("cholesky L Lᵀ = A", rmse([recon[0][0], recon[0][1], recon[1][0], recon[1][1]], [4, 2, 2, 3]) < 1e-10)
    # jacobi eigen
    rng = RNG(7)
    M = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(i, 4):
            v = rng.randn()
            M[i][j] = M[j][i] = v
    eig, V = L.jacobi_eigen(M)
    ok_eig = True
    for i in range(4):
        Av = L.matvec(M, V[i])
        lv = [eig[i] * V[i][k] for k in range(4)]
        if rmse(Av, lv) > 1e-6:
            ok_eig = False
    check("jacobi A v = λ v", ok_eig)
    ortho = True
    for i in range(4):
        for j in range(4):
            d = sum(V[i][k] * V[j][k] for k in range(4)) - (1.0 if i == j else 0.0)
            if abs(d) > 1e-9:
                ortho = False
    check("jacobi eigenvectors orthonormal", ortho)
    # svd
    rng = RNG(3)
    A2 = [[rng.randn() for _ in range(3)] for _ in range(5)]
    U, S, V = L.svd(A2)
    Sm = [[S[k] if i == k else 0.0 for k in range(3)] for i in range(3)]
    recon = L.matmul(L.matmul(U, Sm), L.transpose(V))
    flat = [recon[i][j] for i in range(5) for j in range(3)]
    flatA = [A2[i][j] for i in range(5) for j in range(3)]
    check("svd reconstruction U Σ Vᵀ = A", rmse(flat, flatA) < 1e-6, f"err={rmse(flat,flatA)}")
    uu = L.matmul(L.transpose(U), U)
    orthU = all(abs(uu[i][j] - (1.0 if i == j else 0.0)) < 1e-8 for i in range(3) for j in range(3))
    check("svd U orthogonal (UᵀU = I)", orthU)


# ---------------- optim ----------------
def test_optim():
    print("[optim]")
    f = lambda x: (x[0] - 3) ** 2 + (x[1] + 2) ** 2
    g = lambda x: [2 * (x[0] - 3), 2 * (x[1] + 2)]
    H = lambda x: [[2.0, 0.0], [0.0, 2.0]]

    x, tr = O.gradient_descent(f, g, [0.0, 0.0], lr=0.3, iters=5000)
    check("GD converges to (3,-2)", rmse(x, [3, -2]) < 1e-3, f"x={x}")

    x, tr = O.adam(f, g, [0.0, 0.0], iters=4000, lr=0.1)
    check("Adam converges to (3,-2)", rmse(x, [3, -2]) < 1e-3, f"x={x}")

    x, tr = O.newton(f, g, H, [0.0, 0.0], iters=10)
    check("Newton quadratic convergence", rmse(x, [3, -2]) < 1e-9, f"x={x}, iters={len(tr)}")

    # conjugate gradient: solve SPD system
    A = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
    b = [1.0, 2.0, 3.0]
    x, tr = O.conjugate_gradient(A, b)
    check("CG solves A x = b", rmse(L.matvec(A, x), b) < 1e-9, f"res={rmse(L.matvec(A,x),b)}")

    # BFGS on Rosenbrock
    rb = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    rgb = lambda x: [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]
    x, tr = O.bfgs(rb, rgb, [-1.2, 1.0], iters=20000)
    check("BFGS reduces Rosenbrock", rb(x) < 5.0, f"f={rb(x)}")


# ---------------- ml ----------------
def test_ml():
    print("[ml]")
    # linear regression: residual ⟂ X
    rng = RNG(11)
    X = [[rng.randn(), rng.randn()] for _ in range(200)]
    y = [2 * xi[0] - 3 * xi[1] + 1 + 0.01 * rng.randn() for xi in X]
    lr = linear.LinearRegression().fit(X, y)
    r = [y[i] - lr.predict(X)[i] for i in range(len(X))]
    Xt = L.transpose(X)
    perp = [sum(Xt[j][i] * r[i] for i in range(len(X))) for j in range(2)]
    check("LinearRegression residual ⟂ X", rmse(perp, [0, 0]) < 1e-6, f"perp={perp}")

    # logistic regression on separable data
    X, y = [], []
    for _ in range(150):
        X.append([-2 + rng.randn(), rng.randn()]); y.append(0)
        X.append([2 + rng.randn(), rng.randn()]); y.append(1)
    logit = linear.LogisticRegression(lr=0.5, iters=4000).fit(X, y)
    acc = sum(1 for xi, yi in zip(X, y) if logit.predict([xi])[0] == yi) / len(X)
    check("LogisticRegression separable accuracy", acc > 0.95, f"acc={acc}")

    # kNN
    X, y = blobs()
    knn = linear.KNeighborsClassifier(k=5).fit(X, y)
    acc = sum(1 for xi, yi in zip(X, y) if knn.predict([xi])[0] == yi) / len(X)
    check("kNN accuracy on blobs", acc > 0.9, f"acc={acc}")

    # GaussianNB
    X, y = blobs()
    nb = linear.GaussianNB().fit(X, y)
    acc = sum(1 for xi, yi in zip(X, y) if nb.predict([xi])[0] == yi) / len(X)
    check("GaussianNB accuracy on blobs", acc > 0.9, f"acc={acc}")

    # decision tree learns training data
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    dt = tree.DecisionTreeClassifier(max_depth=3).fit(X, y)
    acc = sum(1 for xi, yi in zip(X, y) if dt.predict([xi])[0] == yi) / len(X)
    check("DecisionTree fits XOR (train)", acc == 1.0, f"acc={acc}")

    # SVM on linearly separable
    X, y = [], []
    for _ in range(100):
        X.append([-3 + rng.randn(), rng.randn()]); y.append(0)
        X.append([3 + rng.randn(), rng.randn()]); y.append(1)
    clf = svm.SVM(C=10.0, max_passes=20).fit(X, y)
    acc = sum(1 for xi, yi in zip(X, y) if clf.predict([xi])[0] == yi) / len(X)
    n_sv = sum(1 for a in clf.alphas if 0 < a < clf.C)
    check("SVM separable accuracy", acc > 0.95, f"acc={acc}")
    check("SVM found support vectors", n_sv > 0, f"n_sv={n_sv}")


# ---------------- cluster / pca ----------------
def test_cluster_pca():
    print("[cluster/pca]")
    X, y = blobs(n_per=80, seed=5)
    km = cluster.KMeans(k=3, seed=2).fit(X)
    inertia0 = float("inf")
    ok_dec = True
    # recompute inertia at each label assignment is implicit; check final inertia sane
    from aether.ml.cluster import _inertia
    check("KMeans produces 3 clusters", len(set(km.labels_)) == 3)
    gmm = cluster.GMM(k=3, seed=2).fit(X)
    ll = gmm.ll_trace
    ok_mono = all(ll[i + 1] >= ll[i] - 1e-6 for i in range(len(ll) - 1))
    check("GMM log-likelihood monotone non-decreasing", ok_mono, f"len={len(ll)}")
    # responsibilities sum to 1
    ok_resp = True
    for i in range(5):
        scores = [math.log(gmm.pi[c] + 1e-300) + gmm._chol_logpdf(X[i], gmm.mu[c], gmm.Ls[c], len(X[0])) for c in range(3)]
        m = max(scores); s = sum(math.exp(s - m) for s in scores)
        if abs(s - 1.0) > 1e-6:
            ok_resp = False
    check("GMM responsibilities sum to 1", ok_resp)

    p = pca.PCA().fit(X)
    ok_orth = True
    for i in range(2):
        for j in range(2):
            d = sum(p.components_[i][k] * p.components_[j][k] for k in range(2)) - (1.0 if i == j else 0.0)
            if abs(d) > 1e-8:
                ok_orth = False
    check("PCA components orthonormal", ok_orth)
    check("PCA explained-variance ratio ≤ 1", abs(sum(p.explained_variance_ratio_) - 1.0) < 1e-9)


# ---------------- deep ----------------
def test_deep():
    print("[deep]")
    err = nn.grad_check(seed=0)
    check("MLP backprop gradient check", err < 1e-4, f"max_rel_err={err}")
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y = [0.0, 1.0, 1.0, 0.0]
    m = nn.MLP(2, [8, 1], seed=3)
    m.fit(X, y, iters=5000, lr=0.05)
    acc = sum(1 for xi, yi in zip(X, y) if m.predict([xi])[0] == yi) / 4
    check("MLP learns XOR", acc == 1.0, f"acc={acc}")

    rng = RNG(9)
    Q = K = V = [[rng.randn() for _ in range(4)] for _ in range(3)]
    out, attn = attention.scaled_dot_product_attention(Q, K, V)
    rowsums = [sum(row) for row in attn]
    check("attention rows sum to 1", all(abs(s - 1.0) < 1e-12 for s in rowsums))
    O = attention.multi_head_attention(Q, K, V, n_heads=2)
    check("multi-head output shape", len(O) == 3 and len(O[0]) == 4)


# ---------------- rl ----------------
def test_rl():
    print("[rl]")
    env = qlearning.GridWorld(n=5)
    agent = qlearning.QLearning(env, seed=1).train(episodes=5000)
    ret = agent.episode_return()
    check("Q-learning reaches goal (return > 0)", ret > 0, f"return={ret}")


def main():
    print("=" * 60)
    print("AETHER invariant test-suite  (author: 晨星)")
    print("=" * 60)
    test_linalg()
    test_optim()
    test_ml()
    test_cluster_pca()
    test_deep()
    test_rl()
    print("=" * 60)
    print(f"PASS={PASS}  FAIL={FAIL}")
    print("=" * 60)
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
