import numpy as np


class MultinomialNaiveBayes:

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):

        self.classes = np.unique(y)

        self.prior = {}
        self.word_prob = {}

        n_samples = X.shape[0]
        n_words = X.shape[1]

        for c in self.classes:

            X_c = X[y == c]

            self.prior[c] = X_c.shape[0] / n_samples

            word_count = X_c.sum(axis=0)

            total_words = word_count.sum()

            self.word_prob[c] = (
                word_count + self.alpha
            ) / (
                total_words + self.alpha * n_words
            )

    def predict(self, X):

        y_pred = []

        for x in X:

            scores = {}

            for c in self.classes:

                log_prior = np.log(
                    self.prior[c]
                )

                log_likelihood = np.sum(
                    x * np.log(
                        self.word_prob[c] + 1e-10
                    )
                )

                scores[c] = (
                    log_prior +
                    log_likelihood
                )

            y_pred.append(
                max(scores, key=scores.get)
            )

        return np.array(y_pred)