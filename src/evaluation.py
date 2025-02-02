from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    roc_curve,
    auc,
    accuracy_score,
)


def evaluate_model(model, X_test, Y_test, model_name=""):
    """
    Evaluate a given model on the test set, returning a dict of metrics.
    """
    predictions = model.predict(X_test)
    acc = accuracy_score(Y_test, predictions)
    rec = recall_score(Y_test, predictions, pos_label=1)
    fpr, tpr, thresholds = roc_curve(Y_test, predictions, pos_label=1)
    model_auc = auc(fpr, tpr)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "recall": rec,
        "auc": model_auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }
