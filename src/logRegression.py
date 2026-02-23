import numpy as np

class CustomLogisticRegression:
    """
    Binary Logistic Regression trained with batch gradient descent.
    Labels: 0 (Safe), 1 (Scam)
    """

    def __init__(self, lr=0.01, epochs=2000, reg_strength=0.0, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.reg_strength = reg_strength  # lambda
        self.verbose = verbose
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)

        # bias 
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.b = float(np.log(p / (1 - p)))

        # class weights 
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        w_pos = n_samples / (2.0 * pos) if pos > 0 else 1.0
        w_neg = n_samples / (2.0 * neg) if neg > 0 else 1.0
        sample_weight = np.where(y == 1, w_pos, w_neg).astype(np.float64)

        max_grad = 20.0
        max_w = 50.0

        for epoch in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)

            error = (p - y) * sample_weight
            dw = (X.T @ error) / n_samples
            db = np.sum(error) / n_samples

            # L2 
            if self.reg_strength > 0:
                dw += self.reg_strength * self.w

            dw = np.clip(dw, -max_grad, max_grad)
            db = float(np.clip(db, -max_grad, max_grad))

            self.w -= self.lr * dw
            self.b -= self.lr * db

            self.w = np.clip(self.w, -max_w, max_w)
            self.b = float(np.clip(self.b, -max_w, max_w))

            if not np.isfinite(self.w).all() or not np.isfinite(self.b):
                raise ValueError("Training diverged: weights became NaN/inf. Try lower lr or higher reg_strength.")

            if self.verbose and epoch % 100 == 0:
                eps = 1e-12
                logloss = -(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
                loss = np.average(logloss, weights=sample_weight)
                if self.reg_strength > 0:
                    loss += 0.5 * self.reg_strength * np.sum(self.w ** 2)
                print(f"epoch={epoch} loss={loss:.4f}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.w + self.b
        p1 = self._sigmoid(z)
        return np.vstack([1 - p1, p1]).T

    def predict(self, X, threshold=0.5):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)