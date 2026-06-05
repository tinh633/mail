import numpy as np
from utils.metrics import compute_evaluation, classification_report


# ─────────────────────────────────────────────────────────────
# Activation
# ─────────────────────────────────────────────────────────────
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ─────────────────────────────────────────────────────────────
# Binary cross-entropy loss
# ─────────────────────────────────────────────────────────────
def _binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )


# ─────────────────────────────────────────────────────────────
# Logistic Regression (viết tay, không dùng sklearn)
# ─────────────────────────────────────────────────────────────
class LogisticRegressionM:
    """
    Binary Logistic Regression huấn luyện bằng Gradient Descent.

    Parameters
    ----------
    learning_rate : float  – bước học (default 0.5)
    n_iters       : int    – số epoch (default 2000)
    threshold     : float  – ngưỡng phân loại (default 0.5)
    """

    def __init__(self, learning_rate=0.5, n_iters=2000, threshold=0.5):
        self.lr        = learning_rate
        self.n_iters   = n_iters
        self.threshold = threshold
        self.w         = None
        self.b         = None

    # ── Huấn luyện ─────────────────────────────────────────
    def fit(self, X, y):
        """
        Gradient Descent trên Binary Cross-Entropy.

        Hỗ trợ cả dense (numpy array) lẫn sparse (scipy) matrix
        nhờ chuyển sang array trước khi tính toán.
        """
        # Scipy sparse → dense để vectorized ops hoạt động
        if hasattr(X, "toarray"):
            X = X.toarray()

        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for i in range(self.n_iters):
            z      = X @ self.w + self.b
            y_pred = _sigmoid(z)

            error = y_pred - y
            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if i % 200 == 0:
                loss = _binary_cross_entropy(y, y_pred)
                print(f"  Epoch {i:>5}/{self.n_iters} | loss = {loss:.4f}")

        return self.w, self.b

    # ── Dự đoán xác suất ───────────────────────────────────
    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return _sigmoid(X @ self.w + self.b)

    # ── Dự đoán nhãn ───────────────────────────────────────
    def predict(self, X, threshold=None):
        threshold = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Đánh giá (trả về dict) ─────────────────────────────
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_true = np.array(y)

        report_data = {}
        for c in [0, 1]:
            TP = int(np.sum((y_pred == c) & (y_true == c)))
            FP = int(np.sum((y_pred == c) & (y_true != c)))
            FN = int(np.sum((y_pred != c) & (y_true == c)))
            support = int(np.sum(y_true == c))

            f1, recall, precision = compute_evaluation(TP, FP, FN)
            report_data[c] = {
                "precision": precision,
                "recall":    recall,
                "f1-score":  f1,
                "support":   support,
            }

        accuracy = float(np.sum(y_pred == y_true) / len(y_true))
        return report_data, accuracy

    # ── In classification report ───────────────────────────
    def classification_Report(self, X, y):
        y_pred = self.predict(X)
        classification_report(y, y_pred, target_names=["ham (0)", "spam (1)"])
