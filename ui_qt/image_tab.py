import threading
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QCheckBox, QFileDialog, QMessageBox, QSlider, QLineEdit, QComboBox, QSizePolicy
from PyQt6.QtCore import Qt

from ui_qt.base_tab import BaseTrainingTab
from modules.model_trainer import get_model, get_incremental_model
from modules.data_loader import load_images_from_folder, LazyImageLoader
from modules.augmentation import get_augmentor
from modules.deep_learning import get_cnn_model, CNNTrainer, prepare_cnn_data, save_cnn_model
from modules.config import DEFAULT_TEST_SIZE, DEFAULT_EPOCHS, DEFAULT_CV_FOLDS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class ImageTrainingTab(BaseTrainingTab):
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
            "Simple CNN 🧠": "model_info_simple_cnn",
            "Deep CNN 🧠": "model_info_deep_cnn",
        }
        super().__init__(lang)

    def setup_config_tab(self):
        layout = QVBoxLayout(self.config_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Dataset
        row = QHBoxLayout()
        btn_folder = QPushButton(self.tr("btn_folder_select"))
        btn_folder.clicked.connect(self.load_folder)
        btn_folder.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.lbl_folder = QLabel(self.tr("lbl_no_folder"))
        row.addWidget(btn_folder)
        row.addWidget(self.lbl_folder)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(QLabel(self.tr("msg_folder_structure")))

        # Models
        layout.addWidget(QLabel(self.tr("header_models")))
        self.models_vars = {}
        models = [
            "SVM", "Random Forest", "KNN", "Logistic Regression", "Decision Tree",
            "Naive Bayes (Gaussian)", "Gradient Boosting", "Simple CNN 🧠", "Deep CNN 🧠"
        ]
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

        # Params
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

        # Output options
        self.create_output_options(layout)
        btn_out_info = QPushButton("ℹ️")
        btn_out_info.clicked.connect(lambda: self.open_info_window(self.tr("lbl_output_options"), self.tr("help_output_options")))
        layout.addWidget(btn_out_info)

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

        batch_row = QHBoxLayout()
        self.chk_batch = QCheckBox(self.tr("lbl_low_memory_mode"))
        batch_row.addWidget(self.chk_batch)
        btn_batch_info = QPushButton("ℹ️")
        btn_batch_info.clicked.connect(lambda: self.open_info_window(self.tr("lbl_low_memory_mode"), self.tr("help_low_memory_mode")))
        batch_row.addWidget(btn_batch_info)
        batch_row.addWidget(QLabel(self.tr("lbl_epoch")))
        self.slider_epoch = QSlider(Qt.Orientation.Horizontal)
        self.slider_epoch.setMinimum(1)
        self.slider_epoch.setMaximum(20)
        self.slider_epoch.setValue(DEFAULT_EPOCHS)
        self.lbl_epoch = QLabel(str(DEFAULT_EPOCHS))
        self.slider_epoch.valueChanged.connect(lambda v: self.lbl_epoch.setText(str(v)))
        batch_row.addWidget(self.slider_epoch)
        batch_row.addWidget(self.lbl_epoch)
        batch_row.addStretch(1)
        layout.addLayout(batch_row)

        aug_row = QHBoxLayout()
        self.chk_aug = QCheckBox(self.tr("lbl_augmentation"))
        aug_row.addWidget(self.chk_aug)
        btn_aug_info = QPushButton("ℹ️")
        btn_aug_info.clicked.connect(lambda: self.open_info_window(self.tr("lbl_augmentation"), self.tr("help_augmentation")))
        aug_row.addWidget(btn_aug_info)
        aug_row.addWidget(QLabel(self.tr("lbl_profile")))
        self.combo_profile = QComboBox()
        self.combo_profile.addItems(["light", "medium", "heavy"])
        self.combo_profile.setMaximumWidth(140)
        self.combo_profile.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        aug_row.addWidget(self.combo_profile)
        aug_row.addWidget(QLabel(self.tr("lbl_multiplier")))
        self.slider_mult = QSlider(Qt.Orientation.Horizontal)
        self.slider_mult.setMinimum(2)
        self.slider_mult.setMaximum(5)
        self.slider_mult.setValue(3)
        self.lbl_mult = QLabel("3x")
        self.slider_mult.valueChanged.connect(lambda v: self.lbl_mult.setText(f"{v}x"))
        aug_row.addWidget(self.slider_mult)
        aug_row.addWidget(self.lbl_mult)
        aug_row.addStretch(1)
        layout.addLayout(aug_row)

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

        optimize = self.chk_opt.isChecked()
        strategy = "all" if self.combo_strategy.currentText() == self.tr("grid_search_all") else "best"
        test_size = self.slider_test.value() / 100
        batch_mode = self.chk_batch.isChecked()
        epochs = self.slider_epoch.value()
        use_aug = self.chk_aug.isChecked()
        aug_profile = self.combo_profile.currentText()
        aug_multiplier = self.slider_mult.value()

        cv_folds = None
        if self.chk_cv.isChecked():
            try:
                cv_folds = int(self.entry_cv.text())
            except ValueError:
                QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_invalid_cv"))
                return

        save_options = self.get_output_options()
        self.tabs.setCurrentWidget(self.results_tab)
        threading.Thread(target=self._training_worker, args=(test_size, optimize, strategy, batch_mode, epochs, use_aug, aug_profile, aug_multiplier, cv_folds, save_options), daemon=True).start()

    def _training_worker(self, test_size, optimize, strategy, batch_mode, epochs, use_aug, aug_profile, aug_multiplier, cv_folds, save_options):
        try:
            X_train = X_test = y_train = y_test = None
            lazy_loader = None
            le = None

            if batch_mode:
                lazy_loader = LazyImageLoader(self.folder_path)
                if len(lazy_loader.files) == 0:
                    raise ValueError(self.tr("msg_no_images"))
                X_train, X_test, y_train, y_test = lazy_loader.get_split(test_size=test_size)
                le = lazy_loader.le
            else:
                X, y = load_images_from_folder(self.folder_path)
                if len(X) == 0:
                    raise ValueError(self.tr("msg_no_images"))
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)

                if use_aug:
                    augmentor = get_augmentor(aug_profile)
                    X_train, y_train = augmentor.augment_dataset(X_train, y_train, multiplier=aug_multiplier)

            selected_models = [n for n, cb in self.models_vars.items() if cb.isChecked()]
            cnn_models = [n for n in selected_models if "CNN" in n]
            sklearn_models = [n for n in selected_models if "CNN" not in n]

            if cnn_models and not batch_mode:
                self._train_cnn_models(cnn_models, test_size, epochs, le)

            models_to_train = {}
            for name in sklearn_models:
                params = self.user_model_params.get(name, {})
                if batch_mode:
                    model = get_incremental_model(name, **params)
                else:
                    model = get_model(name, **params)
                if model is not None:
                    models_to_train[name] = model

            if models_to_train:
                self.run_training_loop(
                    models_to_train,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    extra_data={"label_encoder": le, "model_type": "image"},
                    optimize=optimize,
                    optimize_strategy=strategy,
                    batch_mode=batch_mode,
                    lazy_loader=lazy_loader,
                    epochs=epochs,
                    task_type="classification",
                    cv_folds=cv_folds,
                    save_options=save_options,
                )
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_image_load_error").format(error=e))

    def _train_cnn_models(self, cnn_models, test_size, epochs, label_encoder):
        for model_name in cnn_models:
            clean_name = model_name.replace(" 🧠", "")
            train_loader, val_loader, le, num_classes = prepare_cnn_data(self.folder_path, img_size=64, test_size=test_size, batch_size=32)
            model = get_cnn_model(clean_name, num_classes=num_classes, img_size=64)

            def log_fn(msg, color):
                self.results_manager.log_message(msg, color)

            trainer = CNNTrainer(model, learning_rate=0.001, log_callback=log_fn, stop_check=lambda: self.stop_training_flag)
            history = trainer.train(train_loader, val_loader, epochs=max(epochs, 10))
            final_acc = history["val_acc"][-1] if history["val_acc"] else 0
            self.results_manager.add_result(clean_name, {"Accuracy": final_acc/100, "F1-Score": final_acc/100}, is_classification=True)

            import os, datetime
            save_dir = os.path.join("training_results", f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{clean_name.replace(' ', '_')}.pt")
            save_cnn_model(model, model_path, label_encoder=le)
