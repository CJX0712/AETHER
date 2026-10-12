# AETHER 架构说明

> 本文档描述 AETHER 的每个模块、所复用的世界顶级算法，以及作为「正确性硬标准」的**不变量（invariant）**。
> 设计哲学：**零重写 · 集大成 · 可验证 · 闭环交付**。

---

## 1. 数学底座 `aether/core`

### 1.1 `random.py` — 可复现随机数
复用 **Vigna (2013) 的 splitmix64**：现代标准生成器，能把任意种子扩散为高质量 64 位流。
- 提供 `rand / randn (Box–Muller) / randint / choice / shuffle (Fisher–Yates)`。
- **不变量**：相同种子 → 完全相同的序列（复现实验的底线）。

### 1.2 `linalg.py` — 数值线性代数
| 例程 | 复用方法 | 不变量 |
|------|----------|--------|
| `solve` | 高斯消元 + 部分主元 | `A·x = b` |
| `cholesky` | Cholesky 分解 | `L·Lᵀ = A`（A 对称正定） |
| `jacobi_eigen` | Jacobi 特征值算法（1846，数值稳定，正交特征向量） | `A·v = λ·v` 且 `VᵀV = I` |
| `svd` | 协方差特征分解 + 正交化 | `UΣVᵀ = A`，`UᵀU = I`，`VᵀV = I` |

所有矩阵为行主序的「list of lists」，**无 BLAS 依赖**；SVD 用 Cholesky 稳定求马氏距离（见 `cluster`）。

### 1.3 `optim.py` — 优化器
| 方法 | 复用 | 不变量 |
|------|------|--------|
| `gradient_descent` | 动量梯度下降 | 凸函数收敛到极小值 |
| `adam` | Kingma & Ba (2015) | 自适应矩估计，稳健收敛 |
| `newton` | 牛顿法 + 线搜索 | 二次函数**二次收敛** |
| `conjugate_gradient` | Hestenes & Stiefel (1952) | 在 ≤ n 步内精确求解 SPD 线性系统 |
| `bfgs` | 拟牛顿 + 两循环递归 | 非凸（Rosenbrock）损失显著下降 |
| `line_search` | Armijo 回溯（Nocedal & Wright 2006） | 保证充分下降 |

---

## 2. 机器学习 `aether/ml`

### 2.1 `linear.py`
- **LinearRegression**：正态方程 `w = (XᵀX)⁻¹Xᵀy`（经 Cholesky/微小岭稳定）。
  - *不变量*：残差 ⊥ 设计矩阵，`Xᵀ(y − Xw) ≈ 0`。
- **RidgeRegression**：闭式 `(XᵀX + λI)⁻¹Xᵀy`（Tikhonov 正则）。
- **LogisticRegression**：sigmoid-MLE，Adam 求解。
  - *不变量*：梯度检验；可分数据高准确率。
- **KNeighborsClassifier**：暴力 k-NN（Cover & Hart 1967）。
- **GaussianNB**：条件独立假设下的 MAP 分类器。

### 2.2 `svm.py` — SMO 支持向量机
复用 **Platt (1998) 的序列最小最优化**：每次优化两个拉格朗日乘子，维护 KKT 最优性。
- *不变量*：收敛后支持向量满足 `yᵢ·f(xᵢ) = 1`（0<αᵢ<C），对偶目标最大化。
- 关键工程点：误差缓存 `E = f(x) − y` 必须相对**当前模型**重算，否则偏置 `b` 会发散（已在实现中修正）。

### 2.3 `cluster.py`
- **KMeans**：k-means++ 播种（Arthur & Vassilvitskii 2007）+ Lloyd 迭代。
  - *不变量*：惯性（簇内平方和）**每一步单调下降**。
- **GMM**：期望最大化（Dempster et al. 1977）。
  - *不变量*：对数似然（ELBO）**单调非减**；责任度每行和为 1。密度用 Cholesky 求 `‖Σ⁻¹ᐟ²(x−μ)‖²` 与 `log|Σ|`，数值稳定、无裸矩阵求逆。

### 2.4 `pca.py` — 主成分分析
协方差矩阵 → Jacobi 特征分解 → 按特征值降序排列主轴。
- *不变量*：主成分正交；`explained_variance_ratio_` 之和 ≤ 1 且等于特征值占比。

### 2.5 `tree.py` — CART 决策树
递归二分，按 **Gini 不纯度** 选择分裂（Breiman et al. 1984）。
- 即使根节点增益 ≈ 0（如 XOR），仍继续分裂——因为子节点可进一步分割以降低不纯度。
- *不变量*：可精确拟合训练集可分数据。

---

## 3. 深度学习 `aether/deep`

### 3.1 `nn.py` — 自动微分 + MLP
复用 **Karpathy 的 micrograd 设计**（标量反向自动微分）：
- `Value` 以反向模式（`backward`）实现 `+ − * / pow tanh relu exp`。
- `MLP` 由 `Neuron → Layer` 组合，`fit` 用 **Adam** 训练。
- *不变量*：每个权重的解析梯度与**中心有限差分**交叉验证，rel-err ≈ 1e-6（反向传播正确性的金标准）；MLP 学会 XOR。

### 3.2 `attention.py` — 缩放点积 / 多头注意力
复用 **Vaswani et al. (2017)** Transformer 注意力：
`Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V`，跨头并行。
- *不变量*：每个注意力行是位置上的**概率分布**（行和为 1）。

---

## 4. 强化学习 `aether/rl`

### 4.1 `qlearning.py` — Q-learning
复用 **Watkins & Dayan (1992)** 的离策略时序差分控制：
- 表格 Q 函数 + ε-贪婪探索 + Bellman 最优更新。
- *不变量*：训练后贪心策略到达目标，实现收益等于最优收益。

---

## 5. 验证体系
`tests/test_invariants.py` 用纯 Python 断言上述每一条不变量，**29/29 全绿**，无第三方依赖，CI 友好。
交互演示 `demo/aether_demo.html` 在浏览器内用纯 JS 实时重跑 optim / cluster / pca 三个模块。

---

*作者：晨星 · 站在世界顶级成果之上，而非从零造轮子。*
