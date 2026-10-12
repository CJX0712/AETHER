# AETHER API 参考

纯 Python，零第三方依赖。所有矩阵为「list of lists」（行主序）。

---

## `aether.core.random`
- `RNG(seed: int)` — splitmix64 生成器
  - `rand() -> float` · `randn() -> float` · `randint(lo, hi)` · `choice(n)` · `shuffle(arr)`
- `seeded(seed)` — 便捷构造

## `aether.core.linalg`
- `transpose(A)` · `matmul(A, B)` · `add / sub / scale / identity`
- `solve(A, b)` — 高斯消元（部分主元）
- `cholesky(A)` — 下三角 `L`，`L·Lᵀ = A`
- `jacobi_eigen(A, max_sweeps, tol)` — `(eigenvalues, eigenvectors)`（列为特征向量，正交）
- `svd(A)` — `(U, S, V)`，`A = U·diag(S)·Vᵀ`（薄 SVD）
- `lstsq(X, y, ridge)` — 最小二乘 / 岭回归

## `aether.core.optim`
- `line_search(f, grad, x, d, alpha0, rho, c)` — Armijo 回溯
- `gradient_descent(f, grad, x0, lr, iters, momentum, tol) -> (x, trace)`
- `adam(f, grad, x0, iters, lr, b1, b2, eps, tol) -> (x, trace)`
- `newton(f, grad, hess, x0, iters, tol) -> (x, trace)`
- `conjugate_gradient(A, b, iters, tol) -> (x, trace)`
- `bfgs(f, grad, x0, iters, lr, tol) -> (x, trace)`

## `aether.ml.linear`
- `LinearRegression().fit(X, y).predict(X)`
- `RidgeRegression(alpha).fit(X, y).predict(X)`
- `LogisticRegression(lr, iters).fit(X, y).predict(X) / .predict_proba(X)`
- `KNeighborsClassifier(k).fit(X, y).predict(X)`
- `GaussianNB().fit(X, y).predict(X)`

## `aether.ml.svm`
- `SVM(C, tol, max_passes).fit(X, y).predict(X)` — 线性核 SMO

## `aether.ml.cluster`
- `KMeans(k, seed, max_iter).fit(X)` — 属性 `centers`, `labels_`
- `GMM(k, seed, max_iter, reg).fit(X)` — 属性 `mu`, `pi`, `L`(各分量 Cholesky), `ll_trace`

## `aether.ml.pca`
- `PCA(n_components).fit(X).transform(X) / .fit_transform(X)` — 属性 `components_`, `explained_variance_`, `explained_variance_ratio_`

## `aether.ml.tree`
- `DecisionTreeClassifier(max_depth, min_samples_split).fit(X, y).predict(X)`

## `aether.deep.nn`
- `Value(data, _children, _op)` — 标量自动微分；`+ - * / ** tanh relu exp backward`
- `Neuron(nin, rng, act)` · `Layer(nin, nout, rng, act)` · `MLP(nin, nouts, seed, act)`
  - `MLP.fit(X, y, iters, lr) -> loss` · `MLP.predict(X)` · `MLP.parameters()`
- `grad_check(seed, eps) -> max_rel_err` — 反向传播梯度检验

## `aether.deep.attention`
- `softmax(M)`
- `scaled_dot_product_attention(Q, K, V) -> (out, attn)`
- `multi_head_attention(Q, K, V, n_heads) -> O`

## `aether.rl.qlearning`
- `GridWorld(n, pits, seed)` — `reset / step / n_states / n_actions`
- `QLearning(env, alpha, gamma, eps, seed).train(episodes) -> self`
  - `greedy_policy(s)` · `episode_return(max_steps)`

---

*作者：晨星*
