import numpy as np

class CustomLinearSVC:
    def __init__(self, lr=1e-4, epochs=30, C=1.0, max_w=10.0, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.max_w = max_w
        self.verbose = verbose
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        y2 = np.where(y == 1, 1.0, -1.0)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        # class weights
        pos = np.sum(y2 == 1)
        neg = np.sum(y2 == -1)
        w_pos = n_samples / (2.0 * pos) if pos > 0 else 1.0
        w_neg = n_samples / (2.0 * neg) if neg > 0 else 1.0
        sw = np.where(y2 == 1, w_pos, w_neg).astype(np.float64)

        for ep in range(self.epochs):
            idx = np.random.permutation(n_samples)

            for i in idx:
                xi = X[i]
                yi = y2[i]
                wi = sw[i]

                score = np.dot(self.w, xi) + self.b
                margin = yi * score

                # gradient of 0.5||w||^2 is w
                grad_w = self.w.copy()
                grad_b = 0.0

                # hinge part if margin < 1
                if margin < 1.0:
                    grad_w -= (self.C * wi) * yi * xi
                    grad_b -= (self.C * wi) * yi

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

                # clip to avoid runaway weights
                self.w = np.clip(self.w, -self.max_w, self.max_w)
                self.b = float(np.clip(self.b, -self.max_w, self.max_w))

            if self.verbose and ep % 5 == 0:
                s = self.decision_function(X)
                m = y2 * s
                loss = np.maximum(0.0, 1.0 - m).mean()
                print(f"epoch={ep} hinge_loss={loss:.4f}")

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            s = X @ self.w + self.b
        return np.nan_to_num(s, nan=0.0, posinf=1e6, neginf=-1e6)

    def predict(self, X):
        s = self.decision_function(X)
        return (s >= 0).astype(int)