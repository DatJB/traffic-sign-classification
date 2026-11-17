import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


def classification_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, labels=np.arange(43),
                                 average='macro', zero_division=0)),
    }


def per_class_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=np.arange(43),
                                   output_dict=True, zero_division=0)
    return pd.DataFrame([{'class_id': i, **report[str(i)]} for i in range(43)])
