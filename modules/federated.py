import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def federated_train_classifier(X_train, y_train, X_test, y_test, n_clients: int = 3, epochs: int = 3):
    # Split data into clients
    X_splits = np.array_split(X_train, n_clients)
    y_splits = np.array_split(y_train, n_clients)
    classes = np.unique(y_train)

    model = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", tol=None)

    first_call = True
    for _ in range(epochs):
        for X_c, y_c in zip(X_splits, y_splits):
            if first_call:
                model.partial_fit(X_c, y_c, classes=classes)
                first_call = False
            else:
                model.partial_fit(X_c, y_c)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        "model": model,
        "y_pred": y_pred,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
