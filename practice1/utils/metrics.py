import numpy as np

# compute f-1 score, recall, precision
def compute_evaluation(TP, FP, FN):
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2*(precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score, recall, precision

import numpy as np
import pandas as pd


def evaluate_from_cm(cm, model_name="Model"):

    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    f1 = 2 * precision * recall / (precision + recall)

    tpr = recall

    fpr = fp / (fp + tn)

    print(f"\n===== {model_name} =====")

    print(f"TN : {tn}")
    print(f"FP : {fp}")
    print(f"FN : {fn}")
    print(f"TP : {tp}")

    print()

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"TPR       : {tpr:.4f}")
    print(f"FPR       : {fpr:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TPR": tpr,
        "FPR": fpr,
        "FP": fp,
        "FN": fn
    }
