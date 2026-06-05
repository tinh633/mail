import numpy as np
from utils.metrics import classification_report


# ─────────────────────────────────────────────────────────────
# Multinomial Naive Bayes (viết tay, không dùng sklearn)
# ─────────────────────────────────────────────────────────────
class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes cho bài toán phân loại văn bản.

    Công thức:
        P(y | x) ∝ P(y) * ∏ P(x_i | y)^x_i

    Log-space để tránh underflow:
        log P(y | x) ∝ log P(y) + Σ x_i * log P(x_i | y)

    Parameters
    ----------
    alpha : float – Laplace smoothing (default 1)
    """

    def __init__(self, alpha=1):
        self.alpha      = alpha
        self.classes    = None
        self.prior      = {}       # log prior P(y)
        self.log_prob   = {}       # log likelihood log P(x_i | y)

    # ── Huấn luyện ─────────────────────────────────────────
    def fit(self, X, y):
        """
        X : array-like hoặc scipy sparse, shape (n_samples, n_features)
            Mỗi cell là count (hoặc TF-IDF – dùng như count gần đúng).
        y : array-like, nhãn nguyên.
        """
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.array(X, dtype=float)
        y = np.array(y)

        self.classes  = np.unique(y)
        n_samples, n_words = X.shape

        for c in self.classes:
            X_c = X[y == c]

            # Prior: P(y=c)
            self.prior[c] = X_c.shape[0] / n_samples

            # Word counts với Laplace smoothing
            word_count  = X_c.sum(axis=0)            # (n_words,)
            total_words = word_count.sum()

            self.log_prob[c] = np.log(
                (word_count + self.alpha)
                / (total_words + self.alpha * n_words)
            )

        return self

    # ── Tính log-score cho mỗi class ───────────────────────
    def _compute_log_scores(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.array(X, dtype=float)

        # shape: (n_samples, n_classes)
        log_priors = np.array([np.log(self.prior[c]) for c in self.classes])
        log_likelihoods = np.array(
            [X @ self.log_prob[c] for c in self.classes]
        ).T                                   # (n_samples, n_classes)

        return log_priors + log_likelihoods   # broadcast

    # ── Dự đoán nhãn ───────────────────────────────────────
    def predict(self, X):
        scores = self._compute_log_scores(X)
        idx    = np.argmax(scores, axis=1)
        return self.classes[idx]

    # ── Dự đoán xác suất (softmax trên log-scores) ─────────
    def predict_proba(self, X):
        log_scores = self._compute_log_scores(X)
        # Numerically stable softmax
        log_scores -= log_scores.max(axis=1, keepdims=True)
        proba = np.exp(log_scores)
        return proba / proba.sum(axis=1, keepdims=True)

    # ── In classification report ───────────────────────────
    def classification_Report(self, X, y):
        y_pred = self.predict(X)
        classification_report(y, y_pred, target_names=["ham (0)", "spam (1)"])
