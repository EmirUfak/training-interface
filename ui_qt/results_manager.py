from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QHBoxLayout, QSizePolicy, QScrollArea
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

from modules.languages import get_text
from modules.visualization import create_confusion_matrix_figure, create_feature_importance_figure, create_comparison_figure, create_regression_report


class ResultsManager:
    def __init__(self, results_layout: QVBoxLayout, log_layout: QVBoxLayout,
                 results_scroll: QScrollArea, log_scroll: QScrollArea, lang="tr"):
        self.results_layout = results_layout
        self.log_layout = log_layout
        self.results_scroll = results_scroll
        self.log_scroll = log_scroll
        self.lang = lang

    def tr(self, key):
        return get_text(key, self.lang)

    def log_message(self, text, color):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {color};")
        font = QFont("Consolas", 10)
        lbl.setFont(font)
        self.log_layout.addWidget(lbl)
        self._scroll_to_bottom(self.log_scroll)

    def _scroll_to_bottom(self, scroll: QScrollArea):
        def _do_scroll():
            bar = scroll.verticalScrollBar()
            bar.setValue(bar.maximum())
        QTimer.singleShot(0, _do_scroll)

    def show_model_result(self, name, res, imp_data, y_test, save_dir, save_options=None):
        def _opt(key: str, default: bool = True) -> bool:
            if not save_options:
                return default
            return save_options.get(key, default)

        box = QGroupBox(f"✅ {name}")
        vbox = QVBoxLayout(box)

        is_regression = "r2" in res or "mse" in res
        if is_regression:
            header = QLabel(f"{self.tr('lbl_r2')}: {res['r2']:.4f}")
            header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            vbox.addWidget(header)
            vbox.addWidget(QLabel(f"{self.tr('lbl_r2')}: {res['r2']:.4f} | {self.tr('lbl_mse')}: {res['mse']:.4f} | {self.tr('lbl_mae')}: {res['mae']:.4f}"))

            fig = create_regression_report(y_test, res["y_pred"], model_name=name)
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            vbox.addWidget(canvas)
            if _opt("save_plots"):
                fig.savefig(f"{save_dir}/{name}_regression_report.png")
        else:
            header = QLabel(f"{self.tr('lbl_f1')}: {res['f1']:.4f}")
            header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            vbox.addWidget(header)
            vbox.addWidget(QLabel(f"{self.tr('lbl_accuracy')}: {res['accuracy']:.4f} | {self.tr('lbl_precision')}: {res['precision']:.4f} | {self.tr('lbl_recall')}: {res['recall']:.4f}"))

            tiles = QHBoxLayout()
            tiles.setSpacing(10)

            fig = create_confusion_matrix_figure(y_test, res['y_pred'])
            canvas = FigureCanvas(fig)
            canvas.setFixedSize(320, 320)
            tiles.addWidget(canvas)
            if _opt("save_plots"):
                fig.savefig(f"{save_dir}/{name}_confusion_matrix.png")

            if imp_data:
                importances, feature_names = imp_data
                fig_imp = create_feature_importance_figure(importances, feature_names)
                canvas_imp = FigureCanvas(fig_imp)
                canvas_imp.setFixedSize(320, 320)
                tiles.addWidget(canvas_imp)
                if _opt("save_plots"):
                    fig_imp.savefig(f"{save_dir}/{name}_feature_importance.png")
            tiles.addStretch(1)
            vbox.addLayout(tiles)

        self.results_layout.addWidget(box)
        self._scroll_to_bottom(self.results_scroll)

    def show_comparison(self, results, save_dir, save_options=None):
        def _opt(key: str, default: bool = True) -> bool:
            if not save_options:
                return default
            return save_options.get(key, default)

        df = pd.DataFrame(results)
        metric_col = "F1-Score" if "F1-Score" in df.columns else ("R2 Score" if "R2 Score" in df.columns else df.columns[1])
        df = df.sort_values(by=metric_col, ascending=False)

        box = QGroupBox(self.tr("lbl_model_comparison"))
        vbox = QVBoxLayout(box)
        fig = create_comparison_figure(df, metric=metric_col)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        canvas.setMinimumHeight(300)
        vbox.addWidget(canvas)
        self.results_layout.addWidget(box)
        self._scroll_to_bottom(self.results_scroll)

        if _opt("save_plots"):
            fig.savefig(f"{save_dir}/model_comparison.png")
        if _opt("save_comparison"):
            df.to_csv(f"{save_dir}/results_summary.csv", index=False)

    def show_best_model(self, data, filename, save_options=None):
        box = QGroupBox(self.tr("lbl_best_model"))
        vbox = QVBoxLayout(box)
        if "r2" in data:
            details = f"{self.tr('lbl_model')}: {data['name']}\n{self.tr('lbl_r2')}: {data['r2']:.4f}\n{self.tr('lbl_mse')}: {data['mse']:.4f}"
        else:
            details = f"{self.tr('lbl_model')}: {data['name']}\n{self.tr('lbl_f1')}: {data['f1']:.4f}\n{self.tr('lbl_accuracy')}: {data['acc']:.4f}"
        vbox.addWidget(QLabel(details))
        vbox.addWidget(QLabel(f"{self.tr('lbl_file')}: {filename}"))
        self.results_layout.addWidget(box)
        self._scroll_to_bottom(self.results_scroll)
