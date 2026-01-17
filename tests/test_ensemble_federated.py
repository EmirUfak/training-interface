import numpy as np
from modules.model_trainer import build_voting_classifier, build_voting_regressor, get_model, get_regressor
from modules.federated import federated_train_classifier


def test_voting_classifier_builds():
    models = {
        "SVM": get_model("SVM"),
        "Logistic Regression": get_model("Logistic Regression"),
    }
    voting = build_voting_classifier(models)
    assert voting is not None


def test_federated_classifier_runs():
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    res = federated_train_classifier(X, y, X, y, n_clients=2, epochs=2)
    assert "accuracy" in res
    assert len(res["y_pred"]) == len(y)


def test_voting_regressor_builds():
    models = {
        "Linear Regression": get_regressor("Linear Regression"),
        "Ridge": get_regressor("Ridge"),
    }
    voting = build_voting_regressor(models)
    assert voting is not None
