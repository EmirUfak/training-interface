from matplotlib.figure import Figure
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def create_confusion_matrix_figure(y_test, y_pred):
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"Confusion Matrix", fontsize=10)
    ax.set_ylabel('Gerçek', fontsize=8)
    ax.set_xlabel('Tahmin', fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig

def create_feature_importance_figure(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    top_n = 8
    top_indices = indices[:top_n]
    top_features = feature_names[top_indices]
    top_importances = importances[top_indices]
    
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    sns.barplot(x=top_importances, y=top_features, hue=top_features, ax=ax, palette="viridis", legend=False)
    ax.set_title(f"Önemli Özellikler", fontsize=10)
    ax.set_xlabel("Skor", fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig

def create_comparison_figure(results_df):
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    sns.barplot(x="Model", y="F1-Score", hue="Model", data=results_df, ax=ax, palette="viridis", legend=False)
    ax.set_title("Model Karşılaştırması (F1-Score)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig
