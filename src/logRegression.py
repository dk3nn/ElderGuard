import numpy as np

class CustomLogisticRegression:
    """
    Binary Logistic Regression trained with batch gradient descent.
    Labels: 0 (Safe), 1 (Scam)
    """

    def __init__(self, lr=0.1, epochs=1000, reg_strength=0.0, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.reg_strength = reg_strength  # L2 regularization lambda
        self.verbose = verbose
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        # Prevent overflow for large magnitude z
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        """
        X: (n_samples, n_features) dense numpy array
        y: (n_samples,) numpy array of 0/1
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for epoch in range(self.epochs):
            # Linear scores
            z = X @ self.w + self.b

            # Probabilities
            p = self._sigmoid(z)

            # Gradients (average over samples)
            error = (p - y)  # (n_samples,)
            dw = (X.T @ error) / n_samples
            db = np.sum(error) / n_samples

            # L2 regularization on weights (not bias)
            if self.reg_strength > 0:
                dw += (self.reg_strength / n_samples) * self.w

            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if self.verbose and epoch % 100 == 0:
                # Log loss
                eps = 1e-12
                loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                print(f"epoch={epoch} loss={loss:.4f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.w + self.b
        p1 = self._sigmoid(z)
        # Return shape like sklearn: [P(class0), P(class1)]
        return np.vstack([1 - p1, p1]).T

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)