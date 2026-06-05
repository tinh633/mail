import numpy as np
from utils.metrics import classification_report


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes cho phân loại văn bản.

    Fix so với bản cũ
    ------------------
    1. Bern-style fallback : khi X là TF-IDF (float 0-1), nhân feature với
       scale_factor để tăng magnitude trước khi tính log-likelihood,
       tránh tình trạng x_i ≈ 0 → log-likelihood ≈ 0 → prior thắng hoàn toàn.
    2. log_prior thay vì prior : tất cả tính trong log-space tránh underflow.
    3. Vectorized predict : không vòng for qua từng sample.

    Parameters
    ----------
    alpha        : float – Laplace smoothing (default 1)
    scale_factor : float – nhân feature trước log-likelihood (default 10.0)
                           giúp TF-IDF floats (0-1) tạo ra signal đủ mạnh
    """

    def __init__(self, alpha=1, scale_factor=10.0):
        self.alpha        = alpha
        self.scale_factor = scale_factor
        self.classes      = None
        self.log_prior    = {}    # log P(y=c)
        self.log_prob     = {}    # log P(x_i | y=c)

    # ── Huấn luyện ─────────────────────────────────────────
    def fit(self, X, y):
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.array(X, dtype=float)
        y = np.array(y)

        self.classes       = np.unique(y)
        n_samples, n_words = X.shape

        for c in self.classes:
            X_c = X[y == c]

            # Log prior
            self.log_prior[c] = np.log(X_c.shape[0] / n_samples)

            # Word counts với Laplace smoothing
            word_count  = X_c.sum(axis=0)          # (n_words,)
            total_words = word_count.sum()

            self.log_prob[c] = np.log(
                (word_count + self.alpha)
                / (total_words + self.alpha * n_words)
            )

        return self

    # ── Tính log-score (vectorized) ─────────────────────────
    def _log_scores(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.array(X, dtype=float) * self.scale_factor   # ← fix: scale up

        # log_priors: (n_classes,)  log_probs: (n_classes, n_words)
        log_priors = np.array([self.log_prior[c] for c in self.classes])
        log_probs  = np.array([self.log_prob[c]  for c in self.classes])

        # (n_samples, n_classes) = X @ log_probs.T + log_priors
        return X @ log_probs.T + log_priors

    # ── Dự đoán nhãn ───────────────────────────────────────
    def predict(self, X):
        scores = self._log_scores(X)
        idx    = np.argmax(scores, axis=1)
        return self.classes[idx]

    # ── Dự đoán xác suất (softmax trên log-scores) ─────────
    def predict_proba(self, X):
        log_scores = self._log_scores(X)
        log_scores -= log_scores.max(axis=1, keepdims=True)   # stable softmax
        proba = np.exp(log_scores)
        return proba / proba.sum(axis=1, keepdims=True)

    # ── In classification report ───────────────────────────
    def classification_Report(self, X, y):
        y_pred = self.predict(X)
        classification_report(y, y_pred, target_names=["ham (0)", "spam (1)"])