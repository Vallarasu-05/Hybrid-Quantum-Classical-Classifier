"""Evaluation metrics."""
from __future__ import annotations

import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC and confusion matrix."""
    y_true = y.numpy().astype(int)
    prob = torch.sigmoid(logits).numpy()
    y_pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def evaluate(model, X: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    return metrics_from_logits(model(X), y)
