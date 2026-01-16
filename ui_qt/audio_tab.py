import threading
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QCheckBox, QFileDialog, QMessageBox, QSlider, QLineEdit, QComboBox, QSizePolicy
from PyQt6.QtCore import Qt

from ui_qt.base_tab import BaseTrainingTab
from modules.model_trainer import get_model, build_voting_classifier
from modules.data_loader import load_audio_from_folder
from modules.config import DEFAULT_TEST_SIZE, DEFAULT_CV_FOLDS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class AudioTrainingTab(BaseTrainingTab):
    def __init__(self, lang="tr"):
        self.lang = lang
        self.param_config = {
            "SVM": {"C": ("float", self.tr("desc_C")), "kernel": (["linear", "rbf", "poly", "sigmoid"], self.tr("desc_kernel"))},
            "Random Forest": {"n_estimators": ("int", self.tr("desc_n_estimators")), "max_depth": ("int_or_none", self.tr("desc_max_depth")), "min_samples_split": ("int", self.tr("desc_min_samples_split"))},
            "KNN": {"n_neighbors": ("int", self.tr("desc_n_neighbors")), "weights": (["uniform", "distance"], self.tr("desc_weights"))},
            "Logistic Regression": {"C": ("float", self.tr("desc_C")), "max_iter": ("int", self.tr("desc_max_iter"))},
            "Decision Tree": {"max_depth": ("int_or_none", self.tr("desc_max_depth")), "min_samples_split": ("int", self.tr("desc_min_samples_split"))},
            "Naive Bayes (Gaussian)": {"var_smoothing": ("float", self.tr("desc_var_smoothing"))},
            "Gradient Boosting": {"n_estimators": ("int", self.tr("desc_n_estimators")), "learning_rate": ("float", self.tr("desc_learning_rate")), "max_depth": ("int", self.tr("desc_max_depth"))},
        }
        self.model_info_keys = {
            "SVM": "model_info_svm",
            "Random Forest": "model_info_random_forest",
            "KNN": "model_info_knn",
            "Logistic Regression": "model_info_log_reg",
            "Decision Tree": "model_info_decision_tree",
            "Naive Bayes (Gaussian)": "model_info_nb_gaussian",
            "Gradient Boosting": "model_info_gradient_boosting",
            "Ensemble (Voting)": "model_info_ensemble",
        }
        super().__init__(lang)

    def setup_config_tab(self):
        layout = QVBoxLayout(self.config_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        row = QHBoxLayout()
        btn_folder = QPushButton(self.tr("btn_folder_select"))
        btn_folder.clicked.connect(self.load_folder)
        btn_folder.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.lbl_folder = QLabel(self.tr("lbl_no_folder"))
        row.addWidget(btn_folder)
        row.addWidget(self.lbl_folder)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(QLabel(self.tr("msg_folder_structure_audio")))

        layout.addWidget(QLabel(self.tr("header_models")))
        self.models_vars = {}
        models = ["SVM", "Random Forest", "KNN", "Logistic Regression", "Decision Tree", "Naive Bayes (Gaussian)", "Gradient Boosting", "Ensemble (Voting)"]
        models_grid = QGridLayout()
        models_grid.setSizeConstraint(QGridLayout.SizeConstraint.SetFixedSize)
        models_grid.setAlignment(Qt.AlignmentFlag.AlignLeft)
        for idx, name in enumerate(models):
            cb = QCheckBox(name)
            if name == "SVM":
                cb.setChecked(True)
            self.models_vars[name] = cb

            btn_set = None
            if name in self.param_config:
                btn_set = QPushButton("⚙️")
                btn_set.clicked.connect(lambda _, n=name: self.open_settings_window(n))
                btn_set.setFixedSize(28, 28)
                btn_set.setStyleSheet("padding: 4px;")
            btn_info = QPushButton("ℹ️")
            btn_info.clicked.connect(lambda _, n=name: self.open_info_window(n, self.tr(self.model_info_keys[n])))
            btn_info.setFixedSize(28, 28)
            btn_info.setStyleSheet("padding: 4px;")

            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(4)
            cell_layout.addWidget(cb)
            if btn_set:
                cell_layout.addWidget(btn_set)
            cell_layout.addWidget(btn_info)

            row = idx // 2
            col = idx % 2
            models_grid.addWidget(cell, row, col)

        models_container = QWidget()
        models_container.setLayout(models_grid)
        layout.addWidget(models_container)

        layout.addWidget(QLabel(self.tr("header_params")))
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel(self.tr("lbl_test_size")))
        self.slider_test = QSlider(Qt.Orientation.Horizontal)
        self.slider_test.setMinimum(10)
        self.slider_test.setMaximum(50)
        self.slider_test.setValue(int(DEFAULT_TEST_SIZE * 100))
        self.lbl_test_val = QLabel(f"{DEFAULT_TEST_SIZE:.2f}")
        self.slider_test.valueChanged.connect(lambda v: self.lbl_test_val.setText(f"{v/100:.2f}"))
        test_row.addWidget(self.slider_test)
        test_row.addWidget(self.lbl_test_val)
        test_row.addStretch(1)
        layout.addLayout(test_row)

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

        self.btn_train = QPushButton(self.tr("btn_start_training"))
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr("btn_folder_select"))
        if folder:
            self.folder_path = folder
            self.lbl_folder.setText(folder.split("/")[-1])

    def start_training(self):
        if not hasattr(self, "folder_path"):
            QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_select_folder_first"))
            return

        test_size = self.slider_test.value() / 100
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
        threading.Thread(target=self._training_worker, args=(test_size, optimize, strategy, cv_folds, save_options), daemon=True).start()

    def _training_worker(self, test_size, optimize, strategy, cv_folds, save_options):
        try:
            X, y = load_audio_from_folder(self.folder_path)
            if len(X) == 0:
                raise ValueError(self.tr("msg_no_audio"))

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)

            selected_models = [n for n, cb in self.models_vars.items() if cb.isChecked()]
            models_to_train = {}
            base_models = {}
            for name in selected_models:
                if name == "Ensemble (Voting)":
                    continue
                params = self.user_model_params.get(name, {})
                model = get_model(name, **params)
                if model is not None:
                    models_to_train[name] = model
                    base_models[name] = model

            if "Ensemble (Voting)" in selected_models:
                ensemble = build_voting_classifier(base_models)
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
                apply_scaling=True,
                optimize=optimize,
                optimize_strategy=strategy,
                task_type="classification",
                cv_folds=cv_folds,
                save_options=save_options,
            )
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_audio_load_error").format(error=e))
