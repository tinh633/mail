import numpy as np


class LinearSVM:
    def __init__(
        self,
        learning_rate=0.01,
        lambda_param=0.0001,
        epochs=3000,
        print_every=500
    ):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.print_every = print_every

        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y = np.where(y == 0, -1, 1)

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []

        # class weight xử lý lệch lớp
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)

        weight_pos = n_samples / (2 * n_pos)
        weight_neg = n_samples / (2 * n_neg)

        sample_weight = np.where(y == 1, weight_pos, weight_neg)

        for epoch in range(1, self.epochs + 1):

            scores = np.dot(X, self.w) + self.b
            margins = y * scores

            mask = margins < 1

            if np.sum(mask) > 0:
                weighted_y = sample_weight[mask] * y[mask]

                dw = (
                    self.lambda_param * self.w
                    - np.dot(X[mask].T, weighted_y) / n_samples
                )

                db = -np.sum(weighted_y) / n_samples

            else:
                dw = self.lambda_param * self.w
                db = 0

            # learning rate decay
            lr_epoch = self.lr / (1 + 0.001 * epoch)

            self.w -= lr_epoch * dw
            self.b -= lr_epoch * db

            loss = (
                np.mean(sample_weight * np.maximum(0, 1 - margins))
                + self.lambda_param * np.sum(self.w ** 2) / 2
            )

            self.losses.append(loss)

            if epoch % self.print_every == 0:
                print(f"Epoch {epoch}/{self.epochs} - Hinge Loss: {loss:.4f}")

    def decision_function(self, X):
        X = np.array(X)
        return np.dot(X, self.w) + self.b

    def predict(self, X, threshold=0):
        scores = self.decision_function(X)
        return np.where(scores >= threshold, 1, 0)