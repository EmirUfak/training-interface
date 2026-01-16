from matplotlib.figure import Figure
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import os
import logging
import re

logger = logging.getLogger(__name__)

# Windows-1252 smart quotes/control codes -> safe ASCII
_BAD_GLYPHS = {
    "\x91": "'",
    "\x92": "'",
    "\x93": '"',
    "\x94": '"',
    "\x96": "-",
    "\x97": "-",
    "\x85": "...",
    "\x80": "EUR",
    "\x98": "~",
    "\x9c": '"',
}


def _sanitize_label(text):
    if text is None:
        return ""
    s = str(text)
    for bad, repl in _BAD_GLYPHS.items():
        s = s.replace(bad, repl)
    # Remove remaining control chars
    s = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", s)
    return s


def create_confusion_matrix_figure(y_test, y_pred, class_names=None):
    """Confusion Matrix görselleştirmesi."""
    fig = Figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    cm = confusion_matrix(y_test, y_pred)
    
    # Sınıf isimlerini otomatik belirle
    if class_names is None:
        class_names = [str(c) for c in sorted(set(y_test) | set(y_pred))]
    class_names = [_sanitize_label(c) for c in class_names]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title("Confusion Matrix", fontsize=12)
    ax.set_ylabel('Gerçek', fontsize=10)
    ax.set_xlabel('Tahmin', fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def create_feature_importance_figure(importances, feature_names):
    """Özellik önem skoru grafiği."""
    indices = np.argsort(importances)[::-1]
    top_n = min(10, len(importances))
    top_indices = indices[:top_n]
    top_features = np.array([_sanitize_label(f) for f in feature_names[top_indices]])
    top_importances = importances[top_indices]
    
    fig = Figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    sns.barplot(x=top_importances, y=top_features, hue=top_features, ax=ax, palette="viridis", legend=False)
    ax.set_title("Önemli Özellikler (Top 10)", fontsize=12)
    ax.set_xlabel("Skor", fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def create_comparison_figure(results_df, metric="F1-Score"):
    """Model karşılaştırma grafiği."""
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    
    # Metrik sütunu kontrol et
    if metric not in results_df.columns:
        metric = results_df.columns[1] if len(results_df.columns) > 1 else "Score"

    results_df = results_df.copy()
    results_df["Model"] = results_df["Model"].apply(_sanitize_label)
    
    sns.barplot(x="Model", y=metric, hue="Model", data=results_df, ax=ax, palette="viridis", legend=False)
    ax.set_title(f"Model Karşılaştırması ({metric})", fontsize=12)
    ax.set_ylim(0, max(1, results_df[metric].max() * 1.1))
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    fig.tight_layout()
    return fig


def create_full_classification_report(y_true, y_pred, y_proba=None, class_names=None):
    """
    Kapsamlı Classification Report: Confusion Matrix + Metrics + ROC Curve (varsa).
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        y_proba: Olasılık tahminleri (opsiyonel, ROC için)
        class_names: Sınıf isimleri (opsiyonel)
    
    Returns:
        matplotlib Figure
    """
    n_classes = len(np.unique(y_true))
    has_proba = y_proba is not None and n_classes == 2
    
    # Subplot sayısı
    n_cols = 3 if has_proba else 2
    fig = Figure(figsize=(4 * n_cols, 4))
    
    # Sınıf isimlerini belirle
    if class_names is None:
        class_names = [str(c) for c in sorted(set(y_true) | set(y_pred))]
    class_names = [_sanitize_label(c) for c in class_names]
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(1, n_cols, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names)
    ax1.set_title('Confusion Matrix', fontsize=11)
    ax1.set_ylabel('Gerçek', fontsize=9)
    ax1.set_xlabel('Tahmin', fontsize=9)
    ax1.tick_params(labelsize=8)
    
    # 2. Classification Report (metin olarak)
    ax2 = fig.add_subplot(1, n_cols, 2)
    ax2.axis('off')
    
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    except Exception:
        report = classification_report(y_true, y_pred, zero_division=0)
    
    ax2.text(0.05, 0.95, "Classification Report", fontsize=11, fontweight='bold',
             transform=ax2.transAxes, va='top')
    ax2.text(0.05, 0.85, report, fontfamily='monospace', fontsize=8,
             transform=ax2.transAxes, va='top')
    
    # 3. ROC Curve (binary classification için)
    if has_proba:
        ax3 = fig.add_subplot(1, n_cols, 3)
        try:
            # Binary classification
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                proba = y_proba[:, 1]
            else:
                proba = y_proba
            
            fpr, tpr, _ = roc_curve(y_true, proba)
            roc_auc = auc(fpr, tpr)
            
            ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
            ax3.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate', fontsize=9)
            ax3.set_ylabel('True Positive Rate', fontsize=9)
            ax3.set_title('ROC Curve', fontsize=11)
            ax3.legend(loc="lower right", fontsize=9)
            ax3.tick_params(labelsize=8)
        except Exception as e:
            logger.warning(f"ROC curve oluşturulamadı: {e}")
            ax3.text(0.5, 0.5, "ROC Curve\nkullanılamıyor", ha='center', va='center')
    
    fig.tight_layout()
    return fig


def create_regression_report(y_true, y_pred, model_name="Model"):
    """
    Regresyon sonuçları için görselleştirme: Residual Plot + Predicted vs Actual.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin edilen değerler
        model_name: Model ismi
    
    Returns:
        matplotlib Figure
    """
    fig = Figure(figsize=(10, 4))
    
    # 1. Predicted vs Actual
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
    
    # Diagonal çizgi (mükemmel tahmin)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Mükemmel Tahmin')
    
    ax1.set_xlabel('Gerçek Değerler', fontsize=10)
    ax1.set_ylabel('Tahmin Edilen', fontsize=10)
    ax1.set_title(f'{model_name}: Gerçek vs Tahmin', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.tick_params(labelsize=8)
    
    # 2. Residual Plot
    ax2 = fig.add_subplot(1, 2, 2)
    residuals = y_true - y_pred
    
    ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Tahmin Edilen', fontsize=10)
    ax2.set_ylabel('Residual (Hata)', fontsize=10)
    ax2.set_title('Residual Plot', fontsize=11)
    ax2.tick_params(labelsize=8)
    
    fig.tight_layout()
    return fig


def create_roc_curve_figure(y_true, y_proba):
    fig = Figure(figsize=(5, 4))
    ax = fig.add_subplot(111)

    try:
        y_true_arr = np.asarray(y_true)
        if y_true_arr.ndim > 1:
            # Convert one-hot or multi-column labels to class indices
            y_true_arr = np.argmax(y_true_arr, axis=1)

        scores = np.asarray(y_proba)
        if scores.ndim == 2 and scores.shape[1] == 2:
            scores = scores[:, 1]
        fpr, tpr, _ = roc_curve(y_true_arr, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=9)
        ax.set_ylabel('True Positive Rate', fontsize=9)
        ax.set_title('ROC Curve', fontsize=11)
        ax.legend(loc="lower right", fontsize=9)
        ax.tick_params(labelsize=8)
    except Exception as e:
        logger.warning(f"ROC curve oluşturulamadı: {e}")
        ax.text(0.5, 0.5, "ROC Curve\nkullanılamıyor", ha='center', va='center')

    fig.tight_layout()
    return fig

