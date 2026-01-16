from PyQt6.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QMessageBox, QCheckBox, QGroupBox, QGridLayout, QDialog, QFormLayout, QLineEdit, QComboBox, QToolTip, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor

from modules.languages import get_text
from modules.training_manager import TrainingManager
from ui_qt.results_manager import ResultsManager
from ui_qt.training_signals import TrainingSignals


class BaseTrainingTab(QWidget):
    def __init__(self, lang="tr"):
        super().__init__()
        self.lang = lang
        self.user_model_params = {}
        self.param_config = getattr(self, "param_config", {})
        self.save_options = {
            "save_models": True,
            "save_datasets": True,
            "save_vectorizer": True,
            "save_scaler": True,
            "save_extra": True,
            "save_plots": True,
            "save_comparison": True,
            "save_summary": True,
            "save_model_card": True,
            "save_onnx": True,
        }

        self.tabs = QTabWidget()
        self.config_tab = QWidget()
        self.results_tab = QWidget()
        self.tabs.addTab(self.config_tab, self.tr("tab_config"))
        self.tabs.addTab(self.results_tab, self.tr("tab_results"))

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tabs)

        self.setup_config_tab()
        self.setup_results_tab()

        self.signals = TrainingSignals()
        self.signals.log.connect(lambda msg, color: self.results_manager.log_message(msg, color))
        self.signals.result.connect(lambda n, r, i, y, s, o: self.results_manager.show_model_result(n, r, i, y, s, o))
        self.signals.comparison.connect(lambda r, s, o: self.results_manager.show_comparison(r, s, o))
        self.signals.best_model.connect(lambda d, f, o: self.results_manager.show_best_model(d, f, o))
        self.signals.completion.connect(lambda s: QMessageBox.information(self, self.tr("msg_warning"), self.tr("msg_training_complete").format(save_dir=s)))
        self.signals.error.connect(lambda n, e: self.results_manager.log_message(f"❌ {n} {self.tr('msg_error')}: {e}", "red"))

    def tr(self, key):
        return get_text(key, self.lang)

    def setup_config_tab(self):
        pass

    def setup_results_tab(self):
        layout = QVBoxLayout(self.results_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.log_area = QScrollArea()
        self.log_area.setWidgetResizable(True)
        self.log_area.setMinimumHeight(140)
        self.log_area.setMaximumHeight(220)
        log_container = QWidget()
        self.log_layout = QVBoxLayout(log_container)
        self.log_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.log_area.setWidget(log_container)

        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        res_container = QWidget()
        self.results_layout = QVBoxLayout(res_container)
        self.results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_area.setWidget(res_container)

        layout.addWidget(self.log_area, 1)
        layout.addWidget(self.results_area, 3)

        self.results_manager = ResultsManager(
            self.results_layout,
            self.log_layout,
            self.results_area,
            self.log_area,
            lang=self.lang,
        )

        btn_clear = QPushButton(self.tr("btn_clear_results"))
        btn_clear.clicked.connect(self.clear_results)
        layout.addWidget(btn_clear)

    def clear_results(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def open_settings_window(self, model_name):
        if model_name not in self.param_config:
            return
        config = self.param_config[model_name]
        current_params = self.user_model_params.get(model_name, {})

        dlg = QDialog(self)
        dlg.setWindowTitle(f"{model_name} {self.tr('win_settings_title')}")
        layout = QVBoxLayout(dlg)
        form = QFormLayout()
        layout.addLayout(form)

        widgets = {}
        for param, config_val in config.items():
            if isinstance(config_val, tuple) and len(config_val) == 2:
                p_type, p_desc = config_val
            else:
                p_type = config_val
                p_desc = ""

            if isinstance(p_type, list):
                cb = QComboBox()
                cb.addItems([str(x) for x in p_type])
                val = current_params.get(param)
                if val is not None:
                    idx = cb.findText(str(val))
                    if idx >= 0:
                        cb.setCurrentIndex(idx)
                widgets[param] = cb
                form.addRow(param, cb)
            else:
                le = QLineEdit()
                val = current_params.get(param, "")
                if val is not None:
                    le.setText(str(val))
                widgets[param] = le
                form.addRow(param, le)

            if p_desc:
                form.addRow("", QLabel(f"ℹ️ {p_desc}"))

        btn_save = QPushButton(self.tr("btn_save"))
        def _save():
            new_params = {}
            for param, widget in widgets.items():
                if isinstance(widget, QComboBox):
                    val = widget.currentText()
                else:
                    val = widget.text()
                if val.strip() == "":
                    continue
                new_params[param] = val
            self.user_model_params[model_name] = new_params
            dlg.accept()

        btn_save.clicked.connect(_save)
        layout.addWidget(btn_save)
        dlg.exec()

    def open_info_window(self, title, info_text):
        tooltip_html = f"""
        <div style='max-width: 220px; white-space: normal;'>
        <b>{title}</b><br/>{info_text}
        </div>
        """
        QToolTip.showText(QCursor.pos(), tooltip_html, self)

    def create_output_options(self, parent_layout: QVBoxLayout):
        box = QGroupBox(self.tr("lbl_output_options"))
        box.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        grid = QGridLayout(box)
        grid.setAlignment(Qt.AlignmentFlag.AlignLeft)
        grid.setSizeConstraint(QGridLayout.SizeConstraint.SetFixedSize)

        self.var_save_models = QCheckBox(self.tr("opt_save_models"))
        self.var_save_datasets = QCheckBox(self.tr("opt_save_datasets"))
        self.var_save_vectorizer = QCheckBox(self.tr("opt_save_vectorizer"))
        self.var_save_scaler = QCheckBox(self.tr("opt_save_scaler"))
        self.var_save_extra = QCheckBox(self.tr("opt_save_extra"))
        self.var_save_plots = QCheckBox(self.tr("opt_save_plots"))
        self.var_save_comparison = QCheckBox(self.tr("opt_save_comparison"))
        self.var_save_summary = QCheckBox(self.tr("opt_save_summary"))
        self.var_save_model_card = QCheckBox(self.tr("opt_save_model_card"))
        self.var_save_onnx = QCheckBox(self.tr("opt_save_onnx"))

        for cb in [self.var_save_models, self.var_save_datasets, self.var_save_vectorizer, self.var_save_scaler,
                   self.var_save_extra, self.var_save_plots, self.var_save_comparison, self.var_save_summary,
                   self.var_save_model_card, self.var_save_onnx]:
            cb.setChecked(True)

        opts = [
            (self.var_save_models, 0, 0),
            (self.var_save_datasets, 0, 1),
            (self.var_save_vectorizer, 0, 2),
            (self.var_save_scaler, 1, 0),
            (self.var_save_extra, 1, 1),
            (self.var_save_plots, 1, 2),
            (self.var_save_comparison, 2, 0),
            (self.var_save_summary, 2, 1),
            (self.var_save_model_card, 2, 2),
            (self.var_save_onnx, 3, 0),
        ]
        for cb, r, c in opts:
            grid.addWidget(cb, r, c)

        parent_layout.addWidget(box, 0, Qt.AlignmentFlag.AlignLeft)

    def get_output_options(self):
        return {
            "save_models": self.var_save_models.isChecked(),
            "save_datasets": self.var_save_datasets.isChecked(),
            "save_vectorizer": self.var_save_vectorizer.isChecked(),
            "save_scaler": self.var_save_scaler.isChecked(),
            "save_extra": self.var_save_extra.isChecked(),
            "save_plots": self.var_save_plots.isChecked(),
            "save_comparison": self.var_save_comparison.isChecked(),
            "save_summary": self.var_save_summary.isChecked(),
            "save_model_card": self.var_save_model_card.isChecked(),
            "save_onnx": self.var_save_onnx.isChecked(),
        }

    def run_training_loop(self, models, X_train, X_test, y_train, y_test, vectorizer=None,
                          apply_scaling=False, extra_data=None, optimize=False, optimize_strategy="all",
                          batch_mode=False, lazy_loader=None, epochs=5, task_type="classification",
                          cv_folds=None, save_options=None):
        self.stop_training_flag = False
        self.signals.log.emit(self.tr("msg_training_started"), "blue")

        manager = TrainingManager(
            log_callback=lambda msg, color: self.signals.log.emit(msg, color),
            result_callback=lambda n, r, i, y, s: self.signals.result.emit(n, r, i, y, s, save_options),
            comparison_callback=lambda r, s: self.signals.comparison.emit(r, s, save_options),
            best_model_callback=lambda d, f: self.signals.best_model.emit(d, f, save_options),
            completion_callback=lambda s: self.signals.completion.emit(s),
            error_callback=lambda n, e: self.signals.error.emit(n, e),
            stop_check=lambda: self.stop_training_flag,
        )

        manager.run_training_loop(
            models,
            X_train,
            X_test,
            y_train,
            y_test,
            vectorizer,
            apply_scaling,
            extra_data,
            optimize,
            optimize_strategy,
            batch_mode=batch_mode,
            lazy_loader=lazy_loader,
            epochs=epochs,
            task_type=task_type,
            cv_folds=cv_folds,
            save_options=save_options,
        )
