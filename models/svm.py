import numpy as np
from utils.metrics import classification_report


class LinearSVM:
    """
    Linear SVM – Hinge Loss + L2 Regularization, Sub-gradient Descent.

    Loss = (1/N) Σ w_i·max(0, 1 − y_i·(w·x_i+b))  +  (λ/2)‖w‖²

    Fix so với bản cũ
    ------------------
    1. lambda_param  : 0.001 → 0.0001  (0.001 quá lớn trên TF-IDF high-dim:
                                         regularization shrink w về 0 nhanh hơn
                                         gradient học được → model predict all-ham)
    2. learning_rate : 0.01  → 0.05    (với lambda nhỏ hơn, cần lr cao hơn để converge
                                         đủ nhanh trong epochs giới hạn)
    3. epochs        : 3000  → 5000    (TF-IDF ~3000 dims cần nhiều steps hơn để
                                         hyperplane ổn định với high-dim sparse input)
    4. class_weight  : giữ nguyên logic, nhưng áp dụng đúng vào weighted_y
                       để gradient luôn ưu tiên spam khi bị misclassify

    Parameters
    ----------
    learning_rate : float  (default 0.05)
    lambda_param  : float  (default 0.0001)
    epochs        : int    (default 5000)
    """

    def __init__(self, learning_rate=0.05, lambda_param=0.0001, epochs=5000):
        self.lr           = learning_rate
        self.lambda_param = lambda_param
        self.epochs       = epochs
        self.w            = None
        self.b            = None

    # ── Huấn luyện ─────────────────────────────────────────
    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.array(X, dtype=float)
        y = np.array(y)
        y_svm = np.where(y == 0, -1, 1).astype(float)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        # Class-weight: bù imbalance ham/spam
        n_pos = np.sum(y_svm ==  1)
        n_neg = np.sum(y_svm == -1)
        w_pos = n_samples / (2.0 * n_pos)
        w_neg = n_samples / (2.0 * n_neg)
        sample_weight = np.where(y_svm == 1, w_pos, w_neg)

        for epoch in range(self.epochs):
            scores  = X @ self.w + self.b
            margins = y_svm * scores

            mask = margins < 1.0

            if np.any(mask):
                # sample_weight nhân vào gradient: spam bị misclassify
                # tạo ra gradient lớn hơn ham → model học phân biệt spam tốt hơn
                weighted_y = sample_weight[mask] * y_svm[mask]
                dw = (
                    self.lambda_param * self.w
                    - (X[mask].T @ weighted_y) / n_samples
                )
                db = -np.sum(weighted_y) / n_samples
            else:
                dw = self.lambda_param * self.w
                db = 0.0

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 1000 == 0:
                hinge = np.mean(sample_weight * np.maximum(0.0, 1.0 - margins))
                reg   = self.lambda_param * np.dot(self.w, self.w) / 2
                print(f"  Epoch {epoch:>5}/{self.epochs} | loss = {hinge + reg:.4f}")

        return self

    # ── Score thô ──────────────────────────────────────────
    def decision_function(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X @ self.w + self.b

    # ── Dự đoán nhãn {0, 1} ────────────────────────────────
    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, 0)

    # ── In classification report ───────────────────────────
    def classification_Report(self, X, y):
        y_pred = self.predict(X)
        classification_report(y, y_pred, target_names=["ham (0)", "spam (1)"])