import os
import joblib
import logging
import numpy as np
from typing import Any, Dict, Optional
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, cross_val_score
from scipy.sparse import issparse
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

logger = logging.getLogger(__name__)

PARAM_GRIDS = {
    "Naive Bayes": {'alpha': [0.1, 0.5, 1.0, 2.0]},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    "Logistic Regression": {'C': [0.1, 1, 10]},
    "Decision Tree": {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    "Decision Tree (Entropy)": {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    "Gradient Boosting": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5]},
    "KNN": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    # Regression Params
    "Linear Regression": {},
    "Ridge": {'alpha': [0.1, 1.0, 10.0]},
    "Lasso": {'alpha': [0.1, 1.0, 10.0]},
    "SVR": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    "Random Forest Regressor": {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},
    "Gradient Boosting Regressor": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
    "KNN Regressor": {'n_neighbors': [3, 5, 7]}
}

def _clean_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Parametreleri temizle ve tip dönüşümleri uygula."""
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None and v != ""}
    for k, v in clean_kwargs.items():
        if isinstance(v, str):
            if v.lower() == "none":
                clean_kwargs[k] = None
            elif v.isdigit():
                clean_kwargs[k] = int(v)
            else:
                try:
                    clean_kwargs[k] = float(v)
                except ValueError:
                    pass
    return clean_kwargs


def _needs_dense(model) -> bool:
    dense_models = (
        RandomForestClassifier,
        GradientBoostingClassifier,
        DecisionTreeClassifier,
        RandomForestRegressor,
        GradientBoostingRegressor,
        DecisionTreeRegressor,
        KNeighborsClassifier,
        KNeighborsRegressor,
    )
    return isinstance(model, dense_models)


def _ensure_dense_if_needed(model, X):
    if issparse(X) and _needs_dense(model):
        return X.toarray()
    return X


def get_model(name: str, **kwargs):
    """Sınıflandırma modeli döndürür."""
    clean_kwargs = _clean_kwargs(kwargs)

    if name == "Naive Bayes":
        return MultinomialNB(**clean_kwargs)
    elif name == "Naive Bayes (Gaussian)":
        return GaussianNB(**clean_kwargs)
    elif name == "SVM":
        # Probability her zaman True olmalı arayüz için
        clean_kwargs['probability'] = True
        if 'kernel' not in clean_kwargs: clean_kwargs['kernel'] = 'linear'
        return SVC(**clean_kwargs)
    elif name == "Random Forest":
        if 'n_estimators' not in clean_kwargs: clean_kwargs['n_estimators'] = 50
        return RandomForestClassifier(**clean_kwargs)
    elif name == "Logistic Regression":
        if 'max_iter' not in clean_kwargs: clean_kwargs['max_iter'] = 1000
        return LogisticRegression(**clean_kwargs)
    elif name == "Decision Tree":
        if 'criterion' not in clean_kwargs: clean_kwargs['criterion'] = 'gini'
        return DecisionTreeClassifier(**clean_kwargs)
    elif name == "Decision Tree (Entropy)":
        clean_kwargs['criterion'] = 'entropy'
        return DecisionTreeClassifier(**clean_kwargs)
    elif name == "Gradient Boosting":
        return GradientBoostingClassifier(**clean_kwargs)
    elif name == "KNN":
        return KNeighborsClassifier(**clean_kwargs)
    return None

def get_incremental_model(name: str, **kwargs):
    """Batch eğitim (partial_fit) için uygun model döndürür."""
    clean_kwargs = _clean_kwargs(kwargs)

    if name == "Naive Bayes":
        return MultinomialNB(**clean_kwargs)
    elif name == "Naive Bayes (Gaussian)":
        return GaussianNB(**clean_kwargs)
    elif name == "SVM":
        # SVM (Linear) approx for incremental learning
        clean_kwargs.pop('probability', None) # SGD does not use this param same way
        clean_kwargs.pop('kernel', None)
        return SGDClassifier(loss='hinge', **clean_kwargs)
    elif name == "Logistic Regression":
        # LogReg approx for incremental learning
        clean_kwargs.pop('max_iter', None)
        return SGDClassifier(loss='log_loss', **clean_kwargs)
    
    return None

def get_regressor(name: str, **kwargs):
    """Regresyon modeli döndürür."""
    clean_kwargs = _clean_kwargs(kwargs)

    if name == "Linear Regression":
        return LinearRegression(**clean_kwargs)
    elif name == "Ridge":
        return Ridge(**clean_kwargs)
    elif name == "Lasso":
        return Lasso(**clean_kwargs)
    elif name == "SVR":
        return SVR(**clean_kwargs)
    elif name == "Random Forest Regressor":
        if 'n_estimators' not in clean_kwargs: clean_kwargs['n_estimators'] = 50
        return RandomForestRegressor(**clean_kwargs)
    elif name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(**clean_kwargs)
    elif name == "KNN Regressor":
        return KNeighborsRegressor(**clean_kwargs)
    elif name == "Decision Tree Regressor":
        return DecisionTreeRegressor(**clean_kwargs)
    return None

def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    optimize: bool = False,
    model_name: Optional[str] = None,
    task_type: str = "classification",
    cv_folds: Optional[int] = None,
    grid_n_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """Modeli eğitir ve metrikleri döndürür."""
    cv_score = None
    if cv_folds and cv_folds >= 2:
        try:
            effective_folds = cv_folds
            if task_type == "classification":
                unique, counts = np.unique(y_train, return_counts=True)
                min_count = int(np.min(counts)) if len(counts) > 0 else 0
                if min_count < 2:
                    logger.warning("CV atlandı: sınıf sayısı yetersiz.")
                    effective_folds = None
                elif cv_folds > min_count:
                    logger.warning(f"CV fold {cv_folds} -> {min_count} olarak düşürüldü (min sınıf sayısı).")
                    effective_folds = min_count
            else:
                n_samples = len(y_train)
                if cv_folds > n_samples:
                    logger.warning(f"CV fold {cv_folds} -> {n_samples} olarak düşürüldü (örnek sayısı).")
                    effective_folds = n_samples

            if effective_folds and effective_folds >= 2:
                cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42) if task_type == "classification" else KFold(n_splits=effective_folds, shuffle=True, random_state=42)
                scoring = 'f1_weighted' if task_type == "classification" else 'r2'
                X_train_cv = _ensure_dense_if_needed(model, X_train)
                cv_score = float(np.mean(cross_val_score(model, X_train_cv, y_train, cv=cv, scoring=scoring)))
        except Exception as e:
            logger.warning(f"CV hesaplanamadı: {e}")

    X_train_fit = _ensure_dense_if_needed(model, X_train)
    X_test_fit = _ensure_dense_if_needed(model, X_test)

    if optimize and model_name in PARAM_GRIDS:
        try:
            if task_type == "classification":
                unique, counts = np.unique(y_train, return_counts=True)
                min_count = np.min(counts) if len(counts) > 0 else 0
                cv_val = 3
                if min_count < 3:
                    cv_val = int(min_count)
            else:
                cv_val = 3

            if cv_val < 2 and task_type == "classification":
                logger.warning(f"{model_name} optimizasyon atlanıyor (cv={cv_val}).")
                model.fit(X_train_fit, y_train)
            else:
                logger.info(f"{model_name} için Grid Search başlatılıyor (cv={cv_val})...")
                scoring = 'f1_weighted' if task_type == "classification" else 'r2'
                if grid_n_jobs is None:
                    grid_n_jobs = min(4, os.cpu_count() or 1)
                grid = GridSearchCV(model, PARAM_GRIDS[model_name], cv=cv_val, scoring=scoring, n_jobs=grid_n_jobs)
                grid.fit(X_train_fit, y_train)
                model = grid.best_estimator_
                logger.info(f"{model_name} en iyi parametreler: {grid.best_params_}")
        except Exception as e:
            logger.warning(f"{model_name} optimizasyon hatası: {e}. Varsayılan eğitim yapılıyor.")
            model.fit(X_train_fit, y_train)
    else:
        try:
            model.fit(X_train_fit, y_train)
        except Exception as e:
            raise RuntimeError(f"Eğitim hatası (fit): {e}")

    try:
        y_pred = model.predict(X_test_fit)
    except Exception as e:
        raise RuntimeError(f"Tahmin hatası (predict): {e}")

    if task_type == "classification":
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
            "cv_score": cv_score
        }
    else:
        # Regression Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            "model": model,
            "y_pred": y_pred,
            "r2": r2,
            "mse": mse,
            "mae": mae,
            "cv_score": cv_score
        }

def save_model(model, path: str) -> None:
    """Modeli diske kaydeder."""
    joblib.dump(model, path)
