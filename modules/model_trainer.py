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

def get_model(name):
    if name == "Naive Bayes": return MultinomialNB()
    if name == "Naive Bayes (Gaussian)": return GaussianNB()
    if name == "SVM": return SVC(kernel='linear')
    if name == "Random Forest": return RandomForestClassifier(n_estimators=50)
    if name == "Logistic Regression": return LogisticRegression(max_iter=1000)
    if name == "Decision Tree": return DecisionTreeClassifier()
    if name == "Gradient Boosting": return GradientBoostingClassifier()
    if name == "KNN": return KNeighborsClassifier()
    return None

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
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
        "f1": f1
    }

def save_model(model, path):
    joblib.dump(model, path)
