import numpy as np

class ScratchLogisticRegression:
    def __init__(self, lr=0.1, epochs=200, l2=0.0, seed=42):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = X.shape

        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)

            error = p - y
            grad_w = (X.T @ error) / n_samples
            grad_b = error.mean()

            if self.l2 > 0:
                grad_w += (self.l2 / n_samples) * self.w

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict_proba(self, X):
        z = X @ self.w + self.b
        p1 = self._sigmoid(z)
        return np.column_stack([1 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
