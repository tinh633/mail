import numpy as np

def compute_Sigmoid(x):
    return 1/(1 + np.exp(-x))

# Loss function: binary cross entropy
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    # Avoid log(0) by clipping values
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# compute f-1 score, recall, precision
def compute_evaluation(TP, FP, FN):
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2*(precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score, recall, precision

class LogisticRegressionM:
    def __init__(self, learning_rate=0.5, n_iters=2000):
        self.lr=learning_rate
        self.n_iters=n_iters
        self.w=None     
        self.b=None     
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        for i in range(self.n_iters):
            # compute z = X.w + b
            z = np.dot(X, self.w) + self.b
            
            y_pred = compute_Sigmoid(z)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            if i % 500 == 0:
                loss = compute_loss(y, y_pred)
                print(f"epoch = {i:>}/{self.n_iters}: loss = {loss:>4.4f}")
                
        return self.w, self.b
    
    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return compute_Sigmoid(X @ self.w + self.b)
    
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        y_pred = compute_Sigmoid(z)
        
        return (y_pred > 0.2).astype(int)
    
    def evaluate(self,X, y):
        y_pred = np.array(self.predict(X))
        y_true = np.array(y)
        
        report_data = {}
        for c in [0, 1]:
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            support = np.sum(y_true == c)
            
            f1, r, p = compute_evaluation(tp, fp, fn)
            report_data[c] = {"p": p, "r": r, "f1": f1, "sup": support}
            
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        return report_data, accuracy
    
    def classification_Report(self, X, y):
        report_data, accuracy = self.evaluate(X, y)
        n_samples = len(y)

        print(f"\n{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n")
        
        for label in [0, 1]:
            d = report_data[label]
            print(f"{str(label):>15} {d['p']:>10.2f} {d['r']:>10.2f} {d['f1']:>10.2f} {d['sup']:>10}")

        # Macro average
        macro_p = np.mean([report_data[0]['p'], report_data[1]['p']])
        macro_r = np.mean([report_data[0]['r'], report_data[1]['r']])
        macro_f1 = np.mean([report_data[0]['f1'], report_data[1]['f1']])
        
        print(f"{'accuracy':>15} {'':>10} {'':>10} {accuracy:>10.2f} {n_samples:>10}")
        print(f"{'macro avg':>15} {macro_p:>10.2f} {macro_r:>10.2f} {macro_f1:>10.2f} {n_samples:>10.2f}")
        
    