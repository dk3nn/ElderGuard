import numpy as np


class _Node:
    def __init__(self, feature=None, thresh=None, left=None, right=None, value=None):
        self.feature = feature
        self.thresh = thresh
        self.left = left
        self.right = right
        self.value = value


class MyDecisionTree:
    def __init__(self, max_depth=20, min_split=2, max_feats=None):
        self.max_depth = max_depth
        self.min_split = min_split
        self.max_feats = max_feats
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._grow(X, y, 0)
        return self

    def _gini(self, y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y):
        n, d = X.shape
        best = None
        best_g = 1e9

        feats = range(d)
        if self.max_feats is not None:
            feats = np.random.choice(d, self.max_feats, replace=False)

        for f in feats:
            vals = np.unique(X[:, f])
            if len(vals) < 2:
                continue

            cuts = (vals[:-1] + vals[1:]) / 2

            for t in cuts:
                left = y[X[:, f] <= t]
                right = y[X[:, f] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                g = (len(left) * self._gini(left) +
                     len(right) * self._gini(right)) / n

                if g < best_g:
                    best_g = g
                    best = (f, t)

        return best

    def _leaf_value(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _grow(self, X, y, depth):
        if (depth >= self.max_depth or
                len(y) < self.min_split or
                self._gini(y) == 0):
            return _Node(value=self._leaf_value(y))

        split = self._best_split(X, y)
        if split is None:
            return _Node(value=self._leaf_value(y))

        f, t = split
        left_mask = X[:, f] <= t

        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[~left_mask], y[~left_mask], depth + 1)

        return _Node(f, t, left, right)

    def _predict_one(self, x, node):
        while node.value is None:
            if x[node.feature] <= node.thresh:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.root) for x in X])


class CustomRandomForest:
    def __init__(self, n_trees=50, max_depth=20, max_feats="sqrt"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_feats = max_feats
        self.trees = []

    def _feat_count(self, d):
        if self.max_feats == "sqrt":
            return max(1, int(np.sqrt(d)))
        if isinstance(self.max_feats, int):
            return self.max_feats
        return None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape

        self.trees = []

        for _ in range(self.n_trees):
            idx = np.random.randint(0, n, n)
            Xb = X[idx]
            yb = y[idx]

            k = self._feat_count(d)

            tree = MyDecisionTree(
                max_depth=self.max_depth,
                max_feats=k
            )
            tree.fit(Xb, yb)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)

        preds = np.array([t.predict(X) for t in self.trees])

        out = []
        for i in range(preds.shape[1]):
            vals, counts = np.unique(preds[:, i], return_counts=True)
            out.append(vals[np.argmax(counts)])

        return np.array(out)