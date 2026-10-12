"""AETHER command-line entry — a 60-second tour of the core.

Run:  python -m aether

Author: 晨星
"""
import math
from .core import linalg as L
from .core.optim import adam
from .ml.linear import LinearRegression
from .ml.svm import SVM
from .deep.nn import MLP


def demo():
    print("=" * 58)
    print("  AETHER — 世界顶级自校验 AI 核心   作者：晨星")
    print("=" * 58)

    # 1) SVD of a small matrix
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 0.0]]
    U, S, V = L.svd(A)
    Sm = [[S[k] if i == k else 0.0 for k in range(len(S))] for i in range(len(S))]
    recon = L.matmul(L.matmul(U, Sm), L.transpose(V))
    err = math.sqrt(sum((recon[i][j] - A[i][j]) ** 2 for i in range(len(A)) for j in range(len(A[0]))))
    print(f"[linalg] SVD 重构误差 ‖A − UΣVᵀ‖ = {err:.2e}")

    # 2) optimization: Adam vs Rosenbrock
    rb = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    g = lambda x: [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]
    x, tr = adam(rb, g, [-1.2, 1.0], iters=3000, lr=0.05)
    print(f"[optim]   Adam 在 Rosenbrock 上 → f = {rb(x):.4f} (起始 24.2)")

    # 3) linear regression residual ⟂ X
    X = [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 2.0]]
    y = [3.0, 4.0, 7.0, 9.0]
    lr = LinearRegression().fit(X, y)
    r = [y[i] - lr.predict(X)[i] for i in range(len(X))]
    perp = [sum(X[i][j] * r[i] for i in range(len(X))) for j in range(len(X[0]))]
    print(f"[ml]      LinearRegression 残差⊥X = {perp}")

    # 4) MLP learns XOR
    Xx = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    yy = [0.0, 1.0, 1.0, 0.0]
    m = MLP(2, [8, 1], seed=3)
    loss = m.fit(Xx, yy, iters=5000, lr=0.05)
    acc = sum(1 for xi, yi in zip(Xx, yy) if m.predict([xi])[0] == yi) / 4
    print(f"[deep]    MLP 学习 XOR 准确率 = {acc * 100:.0f}%   (loss={loss:.4f})")

    print("=" * 58)
    print("  完整 29 项不变量自测：python tests/test_invariants.py")
    print("  交互演示：          demo/aether_demo.html")
    print("=" * 58)


if __name__ == "__main__":
    demo()
