import threading
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QScrollArea, QCheckBox, QRadioButton, QButtonGroup, QComboBox, QLineEdit, QSizePolicy
from PyQt6.QtCore import Qt
import pandas as pd

from ui_qt.base_tab import BaseTrainingTab
from modules.data_loader import load_categorical_data
from modules.model_trainer import get_model, get_regressor, build_voting_classifier, build_voting_regressor
from modules.config import DEFAULT_CV_FOLDS
from modules.federated import federated_train_classifier
from sklearn.model_selection import train_test_split


class TabularTrainingTab(BaseTrainingTab):
    def __init__(self, lang="tr"):
        self.lang = lang
        self.param_config = {
            "Naive Bayes": {"alpha": ("float", self.tr("desc_alpha"))},
            "Naive Bayes (Gaussian)": {"var_smoothing": ("float", self.tr("desc_var_smoothing"))},
            "SVM": {"C": ("float", self.tr("desc_C")), "kernel": (["linear", "rbf", "poly", "sigmoid"], self.tr("desc_kernel"))},
            "Random Forest": {"n_estimators": ("int", self.tr("desc_n_estimators")), "max_depth": ("int_or_none", self.tr("desc_max_depth")), "min_samples_split": ("int", self.tr("desc_min_samples_split"))},
            "Logistic Regression": {"C": ("float", self.tr("desc_C")), "max_iter": ("int", self.tr("desc_max_iter"))},
            "Decision Tree": {"max_depth": ("int_or_none", self.tr("desc_max_depth")), "min_samples_split": ("int", self.tr("desc_min_samples_split"))},
            "Decision Tree (Entropy)": {"max_depth": ("int_or_none", self.tr("desc_max_depth")), "min_samples_split": ("int", self.tr("desc_min_samples_split"))},
            "Gradient Boosting": {"n_estimators": ("int", self.tr("desc_n_estimators")), "learning_rate": ("float", self.tr("desc_learning_rate")), "max_depth": ("int", self.tr("desc_max_depth"))},
            "KNN": {"n_neighbors": ("int", self.tr("desc_n_neighbors")), "weights": (["uniform", "distance"], self.tr("desc_weights"))},
            "Linear Regression": {},
            "Ridge": {"alpha": ("float", "Regularization Strength")},
            "Lasso": {"alpha": ("float", "Regularization Strength")},
            "SVR": {"C": ("float", self.tr("desc_C")), "kernel": (["linear", "rbf", "poly", "sigmoid"], self.tr("desc_kernel"))},
            "Random Forest Regressor": {"n_estimators": ("int", self.tr("desc_n_estimators")), "max_depth": ("int_or_none", self.tr("desc_max_depth"))},
            "Gradient Boosting Regressor": {"n_estimators": ("int", self.tr("desc_n_estimators")), "learning_rate": ("float", self.tr("desc_learning_rate"))},
            "Decision Tree Regressor": {"max_depth": ("int_or_none", self.tr("desc_max_depth"))},
            "KNN Regressor": {"n_neighbors": ("int", self.tr("desc_n_neighbors"))},
        }
        self.model_info_keys = {
            "Naive Bayes (Gaussian)": "model_info_nb_gaussian",
            "SVM": "model_info_svm",
            "Random Forest": "model_info_random_forest",
            "Logistic Regression": "model_info_log_reg",
            "Decision Tree": "model_info_decision_tree",
            "Decision Tree (Entropy)": "model_info_decision_tree_entropy",
            "Gradient Boosting": "model_info_gradient_boosting",
            "KNN": "model_info_knn",
            "Linear Regression": "model_info_linear_regression",
            "Ridge": "model_info_ridge",
            "Lasso": "model_info_lasso",
            "SVR": "model_info_svr",
            "Random Forest Regressor": "model_info_rf_regressor",
            "Gradient Boosting Regressor": "model_info_gb_regressor",
            "Decision Tree Regressor": "model_info_dt_regressor",
            "KNN Regressor": "model_info_knn_regressor",
            "Ensemble (Voting)": "model_info_ensemble",
        }
        self.classification_models = [
            "Naive Bayes (Gaussian)", "SVM", "Random Forest", "Logistic Regression", "Decision Tree",
            "Decision Tree (Entropy)", "Gradient Boosting", "KNN", "Ensemble (Voting)"
        ]
        self.regression_models = [
            "Linear Regression", "Ridge", "Lasso", "SVR", "Random Forest Regressor",
            "Gradient Boosting Regressor", "Decision Tree Regressor", "KNN Regressor", "Ensemble (Voting)"
        ]
        super().__init__(lang)

    def setup_config_tab(self):
        layout = QVBoxLayout(self.config_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        btn_file = QPushButton(self.tr("btn_file_select"))
        btn_file.clicked.connect(self.select_file)
        btn_file.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.lbl_file = QLabel(self.tr("lbl_no_file"))
        layout.addWidget(btn_file)
        layout.addWidget(self.lbl_file)

        # Column selector
        self.columns_scroll = QScrollArea()
        self.columns_scroll.setWidgetResizable(True)
        self.columns_container = QWidget()
        self.columns_layout = QVBoxLayout(self.columns_container)
        self.columns_scroll.setWidget(self.columns_container)
        layout.addWidget(self.columns_scroll)

        # Task type
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel(self.tr("lbl_task_type")))
        self.combo_task = QComboBox()
        self.combo_task.addItems([self.tr("task_classification"), self.tr("task_regression")])
        self.combo_task.setMaximumWidth(240)
        self.combo_task.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.combo_task.currentTextChanged.connect(self.update_model_list)
        task_row.addWidget(self.combo_task)
        task_row.addStretch(1)
        layout.addLayout(task_row)

        # Models
        layout.addWidget(QLabel(self.tr("header_models")))
        self.models_container = QWidget()
        self.models_layout = QGridLayout(self.models_container)
        self.models_layout.setSizeConstraint(QGridLayout.SizeConstraint.SetFixedSize)
        self.models_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.models_container)
        self.update_model_list(self.tr("task_classification"))

        # Output options
        out_info_row = QHBoxLayout()
        btn_out_info = QPushButton("ℹ️")
        btn_out_info.setFixedSize(28, 28)
        btn_out_info.setStyleSheet("padding: 4px;")
        btn_out_info.clicked.connect(lambda: self.open_info_window(self.tr("lbl_output_options"), self.tr("help_output_options")))
        out_info_row.addWidget(btn_out_info)
        out_info_row.addStretch(1)
        layout.addLayout(out_info_row)
        self.create_output_options(layout)

        self.chk_opt = QCheckBox(self.tr("chk_optimize"))
        layout.addWidget(self.chk_opt)

        strategy_row = QHBoxLayout()
        strategy_row.addWidget(QLabel(self.tr("lbl_grid_search_scope")))
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems([self.tr("grid_search_all"), self.tr("grid_search_best")])
        self.combo_strategy.setMaximumWidth(220)
        self.combo_strategy.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        strategy_row.addWidget(self.combo_strategy)
        strategy_row.addStretch(1)
        layout.addLayout(strategy_row)

        cv_row = QHBoxLayout()
        self.chk_cv = QCheckBox(self.tr("lbl_use_cv"))
        cv_row.addWidget(self.chk_cv)
        btn_cv_info = QPushButton("ℹ️")
        btn_cv_info.clicked.connect(lambda: self.open_info_window(self.tr("lbl_use_cv"), self.tr("help_kfold")))
        cv_row.addWidget(btn_cv_info)
        cv_row.addWidget(QLabel(self.tr("lbl_cv_folds")))
        self.entry_cv = QLineEdit(str(DEFAULT_CV_FOLDS))
        self.entry_cv.setMaximumWidth(80)
        self.entry_cv.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        cv_row.addWidget(self.entry_cv)
        cv_row.addStretch(1)
        layout.addLayout(cv_row)

        self.chk_federated = QCheckBox(self.tr("lbl_federated"))
        layout.addWidget(self.chk_federated)

        self.btn_train = QPushButton(self.tr("btn_start_training"))
        self.btn_train.clicked.connect(self.start_training_thread)
        self.btn_train.setEnabled(False)
        layout.addWidget(self.btn_train)

    def update_model_list(self, task_type):
        while self.models_layout.count():
            item = self.models_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.models_vars = {}
        models = self.regression_models if task_type == self.tr("task_regression") else self.classification_models
        for idx, name in enumerate(models):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            cb = QCheckBox(name)
            row.addWidget(cb)
            self.models_vars[name] = cb
            if name in self.param_config:
                btn_set = QPushButton("⚙️")
                btn_set.clicked.connect(lambda _, n=name: self.open_settings_window(n))
                btn_set.setFixedSize(28, 28)
                btn_set.setStyleSheet("padding: 4px;")
                row.addWidget(btn_set)
            if name in self.model_info_keys:
                btn_info = QPushButton("ℹ️")
                btn_info.clicked.connect(lambda _, n=name: self.open_info_window(n, self.tr(self.model_info_keys[n])))
                btn_info.setFixedSize(28, 28)
                btn_info.setStyleSheet("padding: 4px;")
                row.addWidget(btn_info)
            container = QWidget()
            container.setLayout(row)
            row_idx = idx // 2
            col_idx = idx % 2
            self.models_layout.addWidget(container, row_idx, col_idx)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_file_select"), "", "CSV Files (*.csv)")
        if file_path:
            self.csv_path = file_path
            self.lbl_file.setText(file_path.split("/")[-1])
            self.load_columns()
            self.btn_train.setEnabled(True)

    def load_columns(self):
        try:
            df = pd.read_csv(self.csv_path, nrows=0)
            columns = df.columns.tolist()

            while self.columns_layout.count():
                item = self.columns_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

            self.feature_checks = {}
            self.target_group = QButtonGroup(self)
            self.target_group.setExclusive(True)

            header = QLabel(self.tr("lbl_columns"))
            self.columns_layout.addWidget(header)

            for col in columns:
                row = QHBoxLayout()
                chk = QCheckBox(col)
                chk.setChecked(True)
                self.feature_checks[col] = chk
                row.addWidget(chk)

                radio = QRadioButton(self.tr("col_target"))
                self.target_group.addButton(radio)
                row.addWidget(radio)

                container = QWidget()
                container.setLayout(row)
                self.columns_layout.addWidget(container)

            if columns:
                self.feature_checks[columns[-1]].setChecked(False)
                self.target_group.buttons()[-1].setChecked(True)

        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_file_read_error").format(error=e))

    def start_training_thread(self):
        if not hasattr(self, "csv_path"):
            QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_select_file_first"))
            return

        target_col = None
        for idx, btn in enumerate(self.target_group.buttons()):
            if btn.isChecked():
                target_col = list(self.feature_checks.keys())[idx]

        feature_cols = [col for col, chk in self.feature_checks.items() if chk.isChecked() and col != target_col]
        if not feature_cols:
            QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_select_features"))
            return
        if not target_col:
            QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_select_target"))
            return

        selected_models = [name for name, cb in self.models_vars.items() if cb.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_select_model"))
            return

        is_regression = self.combo_task.currentText() == self.tr("task_regression")
        optimize = self.chk_opt.isChecked()
        strategy = "all" if self.combo_strategy.currentText() == self.tr("grid_search_all") else "best"

        cv_folds = None
        if self.chk_cv.isChecked():
            try:
                cv_folds = int(self.entry_cv.text())
            except ValueError:
                QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_invalid_cv"))
                return

        save_options = self.get_output_options()
        self.tabs.setCurrentWidget(self.results_tab)
        threading.Thread(target=self.start_training, args=(feature_cols, target_col, selected_models, optimize, strategy, is_regression, cv_folds, save_options), daemon=True).start()

    def start_training(self, feature_cols, target_col, selected_models, optimize, strategy, is_regression, cv_folds, save_options):
        try:
            X, y, encoder, label_encoder = load_categorical_data(self.csv_path, feature_cols, target_col, is_regression=is_regression)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models_to_train = {}
            base_models = {}
            for name in selected_models:
                if name == "Ensemble (Voting)":
                    continue
                params = self.user_model_params.get(name, {})
                model = get_regressor(name, **params) if is_regression else get_model(name, **params)
                if model is not None:
                    models_to_train[name] = model
                    base_models[name] = model

            if "Ensemble (Voting)" in selected_models:
                ensemble = build_voting_regressor(base_models) if is_regression else build_voting_classifier(base_models)
                if ensemble is None:
                    QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_ensemble_need_models"))
                else:
                    models_to_train["Ensemble (Voting)"] = ensemble

            self.run_training_loop(
                models_to_train,
                X_train,
                X_test,
                y_train,
                y_test,
                extra_data={
                    "encoder": encoder,
                    "label_encoder": label_encoder,
                    "feature_cols": feature_cols,
                    "target_col": target_col,
                    "model_type": "tabular",
                },
                optimize=optimize,
                optimize_strategy=strategy,
                task_type="regression" if is_regression else "classification",
                cv_folds=cv_folds,
                save_options=save_options,
            )

            if self.chk_federated.isChecked() and not is_regression:
                import os, time
                fed_res = federated_train_classifier(X_train, y_train, X_test, y_test, n_clients=3, epochs=3)
                save_dir = os.path.join("results", f"federated_{time.strftime('%Y%m%d-%H%M%S')}")
                os.makedirs(save_dir, exist_ok=True)
                self.results_manager.show_model_result("Federated SGD", fed_res, None, y_test, save_dir)
            elif self.chk_federated.isChecked() and is_regression:
                QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_federated_classification_only"))
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_training_error").format(error=e))
