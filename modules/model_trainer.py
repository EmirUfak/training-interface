import os
import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

PARAM_GRIDS = {
    "Naive Bayes": {'alpha': [0.1, 0.5, 1.0, 2.0]},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    "Logistic Regression": {'C': [0.1, 1, 10]},
    "Decision Tree": {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    "Decision Tree (Entropy)": {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    "Gradient Boosting": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5]},
    "KNN": {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
}

def get_model(name, **kwargs):
    # Parametreleri temizle (None olanları ve boş stringleri çıkar)
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None and v != ""}
    
    # int/float dönüşümleri
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
                    pass # string olarak kalsın (örn: 'linear')

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

def train_model(model, X_train, y_train, X_test, y_test, optimize=False, model_name=None):
    if optimize and model_name in PARAM_GRIDS:
        try:
            print(f"{model_name} için Grid Search başlatılıyor...")
            grid = GridSearchCV(model, PARAM_GRIDS[model_name], cv=3, scoring='f1_weighted', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"{model_name} en iyi parametreler: {grid.best_params_}")
        except Exception as e:
            print(f"{model_name} optimizasyon hatası: {e}. Varsayılan eğitim yapılıyor.")
            model.fit(X_train, y_train)
    else:
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            raise RuntimeError(f"Eğitim hatası (fit): {e}")

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        raise RuntimeError(f"Tahmin hatası (predict): {e}")

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
        "f1": f1
    }

def save_model(model, path):
    joblib.dump(model, path)
