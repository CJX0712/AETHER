"""CART decision tree classifier (Breiman et al., 1984).

Recursive binary splitting on Gini impurity. Pure-Python, dependency-free.

Author: 晨星
"""


def _gini(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        self.tree_ = self._grow(X, y, 0)
        return self

    def _grow(self, X, y, depth):
        n = len(y)
        if (
            len(set(y)) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or n < self.min_samples_split
        ):
            counts = {}
            for c in y:
                counts[c] = counts.get(c, 0) + 1
            return {"leaf": True, "cls": max(counts, key=counts.get)}

        d = len(X[0])
        best = None
        parent_gini = _gini(y)
        for f in range(d):
            vals = sorted(set(X[i][f] for i in range(n)))
            for t in range(len(vals) - 1):
                thr = (vals[t] + vals[t + 1]) / 2.0
                left = [i for i in range(n) if X[i][f] <= thr]
                right = [i for i in range(n) if X[i][f] > thr]
                if not left or not right:
                    continue
                w = len(left) / n
                gain = parent_gini - w * _gini([y[i] for i in left]) - (1 - w) * _gini([y[i] for i in right])
                if best is None or gain > best[0]:
                    best = (gain, f, thr, left, right)
        if best is None:
            counts = {}
            for c in y:
                counts[c] = counts.get(c, 0) + 1
            return {"leaf": True, "cls": max(counts, key=counts.get)}
        _, f, thr, left, right = best
        node = {
            "leaf": False,
            "feat": f,
            "thr": thr,
            "left": self._grow([X[i] for i in left], [y[i] for i in left], depth + 1),
            "right": self._grow([X[i] for i in right], [y[i] for i in right], depth + 1),
        }
        return node

    def _traverse(self, x, node):
        if node["leaf"]:
            return node["cls"]
        if x[node["feat"]] <= node["thr"]:
            return self._traverse(x, node["left"])
        return self._traverse(x, node["right"])

    def predict(self, X):
        return [self._traverse(x, self.tree_) for x in X]
