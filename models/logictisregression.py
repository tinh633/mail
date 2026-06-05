import numpy as np
from utils.metrics import compute_evaluation, classification_report


def _sigmoid(z):
    # Numerically stable: tránh overflow khi z rất âm/dương
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )


def _binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )


class LogisticRegressionM:
    """
    Binary Logistic Regression – Gradient Descent với class-weight.

    Fix so với bản cũ
    ------------------
    1. learning_rate  : 0.5 → 0.1  (0.5 quá lớn với TF-IDF high-dim, loss không giảm ổn định)
    2. threshold      : 0.5 → 0.3  (data lệch 87% ham → proba spam hiếm khi > 0.5)
    3. class_weight   : thêm mới   (nhân loss gradient theo tỉ lệ nghịch class freq,
                                    ép model chú ý spam thay vì bias về ham)
    4. _sigmoid       : stable version tránh overflow

    Parameters
    ----------
    learning_rate : float  – bước học (default 0.1)
    n_iters       : int    – số epoch  (default 2000)
    threshold     : float  – ngưỡng phân loại (default 0.3)
    class_weight  : str|None – 'balanced' hoặc None (default 'balanced')
    """

    def __init__(self, learning_rate=0.1, n_iters=2000,
                 threshold=0.3, class_weight='balanced'):
        self.lr           = learning_rate
        self.n_iters      = n_iters
        self.threshold    = threshold
        self.class_weight = class_weight
        self.w            = None
        self.b            = None

    # ── Tính sample weight theo class ──────────────────────
    def _get_sample_weights(self, y, n_samples):
        if self.class_weight != 'balanced':
            return np.ones(n_samples)

        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        # w_c = n_samples / (n_classes * n_c)
        w_pos = n_samples / (2.0 * n_pos)
        w_neg = n_samples / (2.0 * n_neg)
        return np.where(y == 1, w_pos, w_neg)

    # ── Huấn luyện ─────────────────────────────────────────
    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()

        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        sample_w = self._get_sample_weights(y, n_samples)

        self.w = np.zeros(n_features)
        self.b = 0.0

        for i in range(self.n_iters):
            z      = X @ self.w + self.b
            y_pred = _sigmoid(z)

            # Gradient có nhân sample_weight để bù imbalance
            error = sample_w * (y_pred - y)          # (n_samples,)
            dw    = (X.T @ error) / n_samples
            db    = np.mean(error)

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
