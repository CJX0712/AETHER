# AETHER · 世界顶级自校验 AI 核心

> **零依赖 · 可复现 · 可验证** — 纯 Python 实现，站在世界顶级数学与算法之上，每个模块以解析不变量交叉验证。

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-≥3.9-3776ab.svg)](https://www.python.org)
[![tests](https://img.shields.io/bge/invariants-29%2F29-green.svg)](tests/test_invariants.py)

**作者：晨星 · 开源地址：** <https://github.com/CJX0712/AETHER>

---

## 为什么是 AETHER

> 「禁止从零造轮子。任何需求，先检索世界已有的最优解——顶级数学、顶级算法、顶级代码——直接复用、组合、超越。」

AETHER 不重新发明数学，而是把已被工程与理论反复验证的**世界顶级方法**用最简洁、最可读、零第三方依赖的方式重新实现，并给每一块装上**可被独立实现交叉验证的不变量（invariant）**。它不是玩具，而是一套「**正确性的硬标准**」：

| 模块 | 复用的世界顶级方法 | 不变量（硬标准） |
|------|------------------|----------------|
| `core.linalg` | Jacobi 特征值分解（1846）、Cholesky、SVD | `A·v = λ·v`、`VᵀV=I`、`L·Lᵀ=A`、`UΣVᵀ=A` |
| `core.optim`  | Adam、Newton、CG、BFGS、Armijo 线搜索 | 损失单调下降 / 二次收敛 / 残差→0 |
| `ml.linear`   | 正态方程、Tikhonov 岭回归、Adam-MLE | 残差⊥X、梯度检验 |
| `ml.svm`      | Platt SMO（1998） | KKT 最优性、对偶目标最大化 |
| `ml.cluster`  | k-means++（2007）、EM（1977） | 惯性单调↓ / 对数似然单调↑ / 责任度和为 1 |
| `ml.tree`     | CART（Breiman 1984） | 训练集可分数据精确拟合 |
| `ml.pca`      | 协方差特征分解 | 主成分正交、解释方差比 |
| `deep.nn`     | 反向传播（micrograd 设计）、Adam | 梯度检验 rel-err ≈ 1e-6、XOR 可学 |
| `deep.attention` | 缩放点积 / 多头注意力（2017） | 注意力行和为 1 |
| `rl.qlearning` | Q-learning（Watkins 1992） | 最优策略到达目标 |

---

## 快速开始

```bash
# 零第三方依赖，仅注册包名
pip install -e .

# 一键运行 29 项不变量自测（全绿）
python tests/test_invariants.py
# 或：python run_tests.py / python -m aether

# 60 秒核心巡览
python -m aether
```

> 无需 NumPy、无需编译、无需 GPU。任何装有 Python ≥ 3.9 的环境都能跑通——这是「可复现」的底线。

---

## 30 秒示例

```python
import aether
from aether.core.optim import adam
from aether.ml.linear import LinearRegression
from aether.deep.nn import MLP

# 一阶优化：在 Rosenbrock 病态曲面上
f  = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
gf = lambda x: [-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2),
                 200*(x[1] - x[0]**2)]
x, trace = adam(f, gf, [-1.2, 1.0]);  print(f(x))   # ≈ 0

# 监督学习：最小二乘（残差自动 ⊥ 设计矩阵）
lr = LinearRegression().fit(X, y);  lr.predict(Xnew)

# 深度学习：MLP 用反向传播 + Adam 学会 XOR
m  = MLP(2, [8, 1], seed=3).fit([[0,0],[0,1],[1,0],[1,1]],
                                [0,1,1,0]);  m.predict([[1,1]])   # 0
```

交互演示（纯 JS，零依赖，浏览器内实时重跑 optim/cluster/pca）：
打开 **`demo/aether_demo.html`**。

---

## 架构

```
AETHER/
├── aether/
│   ├── core/        # 数学底座（站在巨人肩上）
│   │   ├── random.py   # splitmix64 可复现 RNG（Vigna 2013）
│   │   ├── linalg.py   # 矩阵 / 求解 / Cholesky / Jacobi / SVD
│   │   └── optim.py    # GD·Adam·Newton·CG·BFGS·线搜索
│   ├── ml/          # 机器学习（闭环：理论→实现→验证）
│   │   ├── linear.py   # 线性回归 / 岭回归 / 逻辑回归 / kNN / 朴素贝叶斯
│   │   ├── svm.py      # SMO 支持向量机
│   │   ├── cluster.py  # k-means++ / GMM-EM
│   │   ├── pca.py      # 主成分分析
│   │   └── tree.py     # CART 决策树
│   ├── deep/        # 深度学习
│   │   ├── nn.py       # 标量自动微分 + MLP（反向传播）
│   │   └── attention.py# 缩放点积 / 多头注意力
│   └── rl/          # 强化学习
│       └── qlearning.py # Q-learning 网格世界
├── tests/test_invariants.py   # 29 项解析不变量自测（CI 友好）
├── demo/aether_demo.html      # 单文件交互演示（浏览器内实时跑）
├── docs/           # ARCHITECTURE.md / API.md
└── README.md
```

设计原则：
1. **零重写** — 复用世界顶级数学/算法，不重复造轮子。
2. **集大成** — 每一行都站在顶级成果之上；发现更强现成方案即替换。
3. **可验证** — 每个模块都有一条「不变量」，可被独立实现交叉验证（解析解 vs 数值解、对数似然单调、梯度检验…）。
4. **闭环交付** — 开源、可运行、可复现，作者署名 **晨星**。

---

## 测试结果

```
PASS=29  FAIL=0
```

`python tests/test_invariants.py` 覆盖线性代数、优化、监督/无监督学习、深度学习、强化学习共 **29 项**硬标准，全部通过。

---

## 文档

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — 模块、算法与不变量详述
- [API.md](docs/API.md) — 完整 API 参考

---

## 许可

[MIT](LICENSE) · © 2026 晨星 (Morningstar)
