import numpy as np
from modules.model_trainer import get_model, get_regressor, train_model


def test_train_model_classification_cv():
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 1, 0])
    model = get_model("Logistic Regression")

    res = train_model(
        model,
        X,
        y,
        X,
        y,
        optimize=False,
        model_name="Logistic Regression",
        task_type="classification",
        cv_folds=2,
    )

    assert "accuracy" in res
    assert "f1" in res


def test_train_model_regression_cv():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0])
    model = get_regressor("Decision Tree Regressor")

    res = train_model(
        model,
        X,
        y,
        X,
        y,
        optimize=False,
        model_name="Decision Tree Regressor",
        task_type="regression",
        cv_folds=2,
    )

    assert "r2" in res
    assert "mse" in res
