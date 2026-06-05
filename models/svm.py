import numpy as np
from utils.metrics import classification_report


# ─────────────────────────────────────────────────────────────
# Linear SVM – Hinge Loss + L2 Regularization (viết tay)
# ─────────────────────────────────────────────────────────────
class LinearSVM:
    """
    Linear Support Vector Machine huấn luyện bằng Sub-gradient Descent.

    Loss = (1/N) Σ w_i * max(0, 1 − y_i * (w·x_i + b)) + (λ/2) ‖w‖²

    Hỗ trợ class-weight tự động để xử lý mất cân bằng Ham/Spam.

    Parameters
    ----------
    learning_rate : float  – bước học (default 0.01)
    lambda_param  : float  – hệ số L2 regularization (default 0.001)
    epochs        : int    – số vòng lặp (default 3000)
    """

    def __init__(self, learning_rate=0.01, lambda_param=0.001, epochs=3000):
        self.lr           = learning_rate
        self.lambda_param = lambda_param
        self.epochs       = epochs
        self.w            = None
        self.b            = None

    # ── Huấn luyện ─────────────────────────────────────────
    def fit(self, X, y):
        """
        y phải là {0, 1}; tự chuyển sang {-1, +1} nội bộ.
        Hỗ trợ cả dense lẫn sparse matrix.
        """
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.array(X, dtype=float)
        y = np.array(y)
        y_svm = np.where(y == 0, -1, 1).astype(float)   # {0,1} → {-1,+1}

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        # ── Class-weight: bù cho lệch class ─────────────────
        n_pos = np.sum(y_svm ==  1)
        n_neg = np.sum(y_svm == -1)
        weight_pos = n_samples / (2 * n_pos)
        weight_neg = n_samples / (2 * n_neg)
        sample_weight = np.where(y_svm == 1, weight_pos, weight_neg)

        for epoch in range(self.epochs):
            scores  = X @ self.w + self.b        # (n_samples,)
            margins = y_svm * scores             # y_i * (w·x_i + b)

            # Chỉ cập nhật những mẫu vi phạm margin ( margin < 1 )
            mask = margins < 1.0

            if np.any(mask):
                weighted_y = sample_weight[mask] * y_svm[mask]
                dw = (
                    self.lambda_param * self.w
                    - (X[mask].T @ weighted_y) / n_samples
                )
                db = -np.sum(weighted_y) / n_samples
            else:
                # Tất cả mẫu đều ngoài margin: chỉ regularize
                dw = self.lambda_param * self.w
                db = 0.0

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 500 == 0:
                hinge = np.mean(sample_weight * np.maximum(0.0, 1.0 - margins))
                reg   = self.lambda_param * np.dot(self.w, self.w) / 2
                print(f"  Epoch {epoch:>5}/{self.epochs} | loss = {hinge + reg:.4f}")

        return self

    # ── Score thô (khoảng cách tới siêu phẳng) ─────────────
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
