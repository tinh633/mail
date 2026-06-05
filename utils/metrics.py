import numpy as np


# ─────────────────────────────────────────────
# Tính Precision, Recall, F1 cho một class
# ─────────────────────────────────────────────
def compute_evaluation(TP, FP, FN):
    """
    Trả về (f1_score, recall, precision) cho một class cụ thể.
    Trả về 0 khi mẫu số = 0 để tránh ZeroDivisionError.
    """
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1_score  = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return f1_score, recall, precision


# ─────────────────────────────────────────────
# Confusion matrix viết tay (2 class: 0 / 1)
# ─────────────────────────────────────────────
def confusion_matrix(y_true, y_pred, labels=None):
    """
    Trả về numpy array shape (n_classes, n_classes).
    confusion_matrix[i][j] = số mẫu thực tế là class i, dự đoán là class j.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    n = len(labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}

    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[label_to_idx[t]][label_to_idx[p]] += 1

    return matrix


# ─────────────────────────────────────────────
# Classification report viết tay
# ─────────────────────────────────────────────
def classification_report(y_true, y_pred, labels=None, target_names=None):
    """
    In và trả về dict report gồm precision, recall, f1, support
    cho từng class, cùng accuracy và macro avg.

    Parameters
    ----------
    y_true       : array-like, nhãn thực tế
    y_pred       : array-like, nhãn dự đoán
    labels       : list các class cần report (mặc định tự detect)
    target_names : tên hiển thị tương ứng với labels
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

    if target_names is None:
        target_names = [str(lbl) for lbl in labels]

    report_data = {}

    for lbl, name in zip(labels, target_names):
        TP = int(np.sum((y_pred == lbl) & (y_true == lbl)))
        FP = int(np.sum((y_pred == lbl) & (y_true != lbl)))
        FN = int(np.sum((y_pred != lbl) & (y_true == lbl)))
        support = int(np.sum(y_true == lbl))

        f1, recall, precision = compute_evaluation(TP, FP, FN)
        report_data[name] = {
            "precision": precision,
            "recall":    recall,
            "f1-score":  f1,
            "support":   support,
        }

    # Accuracy
    accuracy  = float(np.sum(y_true == y_pred) / len(y_true))
    n_samples = len(y_true)

    # Macro average
    macro_p  = float(np.mean([report_data[n]["precision"] for n in target_names]))
    macro_r  = float(np.mean([report_data[n]["recall"]    for n in target_names]))
    macro_f1 = float(np.mean([report_data[n]["f1-score"]  for n in target_names]))

    # ── In report ──────────────────────────────────────────
    col_width = max(len(n) for n in target_names) + 2
    header = f"\n{'':<{col_width}} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n"
    print(header)

    for name in target_names:
        d = report_data[name]
        print(
            f"{name:<{col_width}} "
            f"{d['precision']:>10.2f} "
            f"{d['recall']:>10.2f} "
            f"{d['f1-score']:>10.2f} "
            f"{d['support']:>10}"
        )

    print()
    print(f"{'accuracy':<{col_width}} {'':>10} {'':>10} {accuracy:>10.2f} {n_samples:>10}")
    print(
        f"{'macro avg':<{col_width}} "
        f"{macro_p:>10.2f} "
        f"{macro_r:>10.2f} "
        f"{macro_f1:>10.2f} "
        f"{n_samples:>10}"
    )
    print()

    # Trả về dict đầy đủ (tiện so sánh sau)
    report_data["accuracy"]   = accuracy
    report_data["macro avg"]  = {
        "precision": macro_p,
        "recall":    macro_r,
        "f1-score":  macro_f1,
        "support":   n_samples,
    }
    return report_data
