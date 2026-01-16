import threading
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QLineEdit, QFileDialog, QMessageBox, QSizePolicy
from PyQt6.QtCore import Qt
import pandas as pd

from ui_qt.base_tab import BaseTrainingTab
from modules.model_trainer import get_model
from modules.data_loader import load_and_vectorize_text, get_stop_words
from modules.config import DEFAULT_MAX_FEATURES, DEFAULT_NGRAM, DEFAULT_TEST_SIZE, DEFAULT_CV_FOLDS


class TextTrainingTab(BaseTrainingTab):
    def __init__(self, lang="tr"):
        self.lang = lang
        self.param_config = {
            "Naive Bayes": {"alpha": ("float", self.tr("desc_alpha"))},
            "SVM": {
                "C": ("float", self.tr("desc_C")),
                "kernel": (["linear", "rbf", "poly", "sigmoid"], self.tr("desc_kernel"))
            },
            "Random Forest": {
                "n_estimators": ("int", self.tr("desc_n_estimators")),
                "max_depth": ("int_or_none", self.tr("desc_max_depth")),
                "min_samples_split": ("int", self.tr("desc_min_samples_split"))
            },
            "Logistic Regression": {
                "C": ("float", self.tr("desc_C")),
                "max_iter": ("int", self.tr("desc_max_iter"))
            },
            "Decision Tree": {
                "max_depth": ("int_or_none", self.tr("desc_max_depth")),
                "min_samples_split": ("int", self.tr("desc_min_samples_split"))
            },
            "Gradient Boosting": {
                "n_estimators": ("int", self.tr("desc_n_estimators")),
                "learning_rate": ("float", self.tr("desc_learning_rate")),
                "max_depth": ("int", self.tr("desc_max_depth"))
            },
            "KNN": {
                "n_neighbors": ("int", self.tr("desc_n_neighbors")),
                "weights": (["uniform", "distance"], self.tr("desc_weights"))
            }
        }
        self.model_info_keys = {
            "Naive Bayes": "model_info_naive_bayes",
            "SVM": "model_info_svm",
            "Random Forest": "model_info_random_forest",
            "Logistic Regression": "model_info_log_reg",
            "Decision Tree": "model_info_decision_tree",
            "Gradient Boosting": "model_info_gradient_boosting",
            "KNN": "model_info_knn",
        }
        super().__init__(lang)

    def setup_config_tab(self):
        layout = QVBoxLayout(self.config_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Dataset
        ds_row = QHBoxLayout()
        btn_file = QPushButton(self.tr("btn_file_select"))
        btn_file.clicked.connect(self.load_csv)
        btn_file.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.lbl_file = QLabel(self.tr("lbl_no_file"))
        ds_row.addWidget(btn_file)
        ds_row.addWidget(self.lbl_file)
        ds_row.addStretch(1)
        layout.addLayout(ds_row)

        cols_row = QHBoxLayout()
        cols_row.addWidget(QLabel(self.tr("col_text")))
        self.combo_text = QComboBox()
        self.combo_text.setMaximumWidth(260)
        self.combo_text.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.combo_text.addItem(self.tr("msg_select_file_first"))
        cols_row.addWidget(self.combo_text)
        cols_row.addWidget(QLabel(self.tr("col_target")))
        self.combo_label = QComboBox()
        self.combo_label.setMaximumWidth(260)
        self.combo_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.combo_label.addItem(self.tr("msg_select_file_first"))
        cols_row.addWidget(self.combo_label)
        cols_row.addStretch(1)
        layout.addLayout(cols_row)

        # Models
        layout.addWidget(QLabel(self.tr("header_models")))
        self.models_vars = {}
        models_grid = QGridLayout()
        models_grid.setSizeConstraint(QGridLayout.SizeConstraint.SetFixedSize)
        models_grid.setAlignment(Qt.AlignmentFlag.AlignLeft)
        models = ["Naive Bayes", "SVM", "Random Forest", "Logistic Regression", "Decision Tree", "Gradient Boosting", "KNN"]
        for idx, name in enumerate(models):
            cb = QCheckBox(name)
            if name in ["Naive Bayes", "SVM"]:
                cb.setChecked(True)
            self.models_vars[name] = cb

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
            cell_layout.addWidget(btn_set)
            cell_layout.addWidget(btn_info)

            row = idx // 2
            col = idx % 2
            models_grid.addWidget(cell, row, col)

        models_container = QWidget()
        models_container.setLayout(models_grid)
        layout.addWidget(models_container)

        # Params
        layout.addWidget(QLabel(self.tr("header_params")))
        params_row = QHBoxLayout()
        params_row.addWidget(QLabel(self.tr("lbl_test_size")))
        self.slider_test = QSlider(Qt.Orientation.Horizontal)
        self.slider_test.setMinimum(10)
        self.slider_test.setMaximum(50)
        self.slider_test.setValue(int(DEFAULT_TEST_SIZE * 100))
        self.lbl_test_val = QLabel(f"{DEFAULT_TEST_SIZE:.2f}")
        self.slider_test.valueChanged.connect(lambda v: self.lbl_test_val.setText(f"{v/100:.2f}"))
        params_row.addWidget(self.slider_test)
        params_row.addWidget(self.lbl_test_val)
        params_row.addStretch(1)
        layout.addLayout(params_row)

        params2 = QHBoxLayout()
        params2.addWidget(QLabel(self.tr("lbl_max_features")))
        self.entry_max = QLineEdit(str(DEFAULT_MAX_FEATURES))
        self.entry_max.setMaximumWidth(120)
        self.entry_max.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        params2.addWidget(self.entry_max)
        params2.addWidget(QLabel(self.tr("lbl_ngram")))
        self.entry_ngram = QLineEdit(DEFAULT_NGRAM)
        self.entry_ngram.setMaximumWidth(120)
        self.entry_ngram.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        params2.addWidget(self.entry_ngram)
        params2.addStretch(1)
        layout.addLayout(params2)

        params3 = QHBoxLayout()
        params3.addWidget(QLabel(self.tr("lbl_stop_words")))
        self.combo_stop = QComboBox()
        self.combo_stop.addItems([self.tr("stop_words_none"), self.tr("stop_words_tr"), self.tr("stop_words_en")])
        self.combo_stop.setMaximumWidth(180)
        self.combo_stop.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        params3.addWidget(self.combo_stop)
        btn_stop_info = QPushButton("ℹ️")
        btn_stop_info.clicked.connect(lambda: self.open_info_window(self.tr("lbl_stop_words"), self.tr("help_stop_words")))
        params3.addWidget(btn_stop_info)
        self.chk_sparse = QCheckBox(self.tr("lbl_sparse_tfidf"))
        self.chk_sparse.setChecked(True)
        params3.addWidget(self.chk_sparse)
        btn_sparse_info = QPushButton("ℹ️")
        btn_sparse_info.clicked.connect(lambda: self.open_info_window(self.tr("lbl_sparse_tfidf"), self.tr("help_sparse_tfidf")))
        params3.addWidget(btn_sparse_info)
        params3.addStretch(1)
        layout.addLayout(params3)

        # Output options
        self.create_output_options(layout)
        btn_out_info = QPushButton("ℹ️")
        btn_out_info.clicked.connect(lambda: self.open_info_window(self.tr("lbl_output_options"), self.tr("help_output_options")))
        layout.addWidget(btn_out_info)

        # Optimize / CV
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

        # Train
        self.btn_train = QPushButton(self.tr("btn_start_training"))
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_file_select"), "", "CSV Files (*.csv)")
        if not file_path:
            return
        self.csv_path = file_path
        self.lbl_file.setText(file_path.split("/")[-1])
        try:
            df = pd.read_csv(file_path)
            columns = df.columns.tolist()
            self.combo_text.clear()
            self.combo_label.clear()
            self.combo_text.addItems(columns)
            self.combo_label.addItems(columns)
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_file_read_error").format(error=e))

    def start_training(self):
        if not hasattr(self, "csv_path"):
            QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_select_file_first"))
            return

        try:
            max_feat = int(self.entry_max.text())
            ngram_txt = self.entry_ngram.text().split(',')
            ngram_range = (int(ngram_txt[0]), int(ngram_txt[1]))
        except ValueError:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_invalid_text_params"))
            return

        stop_words = None
        sw = self.combo_stop.currentText()
        if sw == self.tr("stop_words_tr"):
            stop_words = get_stop_words("turkish")
        elif sw == self.tr("stop_words_en"):
            stop_words = get_stop_words("english")

        cv_folds = None
        if self.chk_cv.isChecked():
            try:
                cv_folds = int(self.entry_cv.text())
            except ValueError:
                QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_invalid_cv"))
                return

        optimize = self.chk_opt.isChecked()
        strategy = "all" if self.combo_strategy.currentText() == self.tr("grid_search_all") else "best"
        test_size = self.slider_test.value() / 100
        use_sparse = self.chk_sparse.isChecked()

        save_options = self.get_output_options()

        self.tabs.setCurrentWidget(self.results_tab)

        threading.Thread(target=self._training_worker, args=(
            self.combo_text.currentText(),
            self.combo_label.currentText(),
            test_size,
            max_feat,
            ngram_range,
            optimize,
            strategy,
            stop_words,
            use_sparse,
            cv_folds,
            save_options
        ), daemon=True).start()

    def _training_worker(self, text_col, label_col, test_size, max_features, ngram_range, optimize, strategy, stop_words, use_sparse, cv_folds, save_options):
        try:
            X_vec, y, vectorizer = load_and_vectorize_text(
                self.csv_path,
                text_col,
                label_col,
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words=stop_words,
                use_sparse=use_sparse,
                preprocess_text=True
            )

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=test_size, random_state=42)

            selected_models = [name for name, cb in self.models_vars.items() if cb.isChecked()]
            models_to_train = {}
            for name in selected_models:
                params = self.user_model_params.get(name, {})
                model = get_model(name, **params)
                if model is not None:
                    models_to_train[name] = model

            self.run_training_loop(
                models_to_train,
                X_train,
                X_test,
                y_train,
                y_test,
                vectorizer=vectorizer,
                optimize=optimize,
                optimize_strategy=strategy,
                task_type="classification",
                cv_folds=cv_folds,
                save_options=save_options,
            )
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_training_error").format(error=e))
