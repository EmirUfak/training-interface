import os
import joblib
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QComboBox, QTextEdit, QScrollArea, QFormLayout, QLineEdit
)
from PyQt6.QtWidgets import QToolTip
from PyQt6.QtGui import QCursor
from PyQt6.QtCore import Qt

from modules.data_loader import load_single_image, load_single_audio
from modules.languages import get_text


class InferenceTab(QWidget):
    def __init__(self, lang="tr"):
        super().__init__()
        self.lang = lang

        self.model = None
        self.scaler = None
        self.vectorizer = None
        self.encoder = None
        self.label_encoder = None
        self.feature_cols = None
        self.tabular_entries = {}
        self.input_file_path = None
        self.batch_csv_path = None

        self.model_type = "text"
        self.setup_ui()

    def tr(self, key):
        return get_text(key, self.lang)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Load area
        load_frame = QWidget()
        load_layout = QVBoxLayout(load_frame)
        load_layout.setSpacing(6)
        load_layout.addWidget(QLabel(self.tr("header_load")))

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel(self.tr("lbl_model_type")))
        self.type_map = {
            self.tr("type_text"): "text",
            self.tr("type_image"): "image",
            self.tr("type_audio"): "audio",
            self.tr("type_tabular"): "tabular",
        }
        self.combo_type = QComboBox()
        self.combo_type.setMaximumWidth(240)
        self.combo_type.addItems(list(self.type_map.keys()))
        self.combo_type.currentTextChanged.connect(self.on_type_change)
        type_row.addWidget(self.combo_type)
        type_row.addStretch(1)
        load_layout.addLayout(type_row)

        model_row = QHBoxLayout()
        self.btn_load_model = QPushButton(self.tr("btn_select_model"))
        self.btn_load_model.clicked.connect(self.load_model_file)
        self.lbl_model_path = QLabel(self.tr("lbl_no_file"))
        model_row.addWidget(self.btn_load_model)
        model_row.addWidget(self.lbl_model_path)
        model_row.addStretch(1)
        load_layout.addLayout(model_row)

        extra_row = QHBoxLayout()
        self.btn_load_extra = QPushButton(self.tr("btn_load_vectorizer"))
        self.btn_load_extra.clicked.connect(self.load_extra_file)
        self.lbl_extra_path = QLabel(self.tr("lbl_not_needed"))
        extra_row.addWidget(self.btn_load_extra)
        extra_row.addWidget(self.lbl_extra_path)
        extra_row.addStretch(1)
        load_layout.addLayout(extra_row)

        self.tabular_files_frame = QWidget()
        tabular_files_layout = QVBoxLayout(self.tabular_files_frame)

        f1 = QHBoxLayout()
        self.btn_load_features = QPushButton(self.tr("btn_load_features"))
        self.btn_load_features.clicked.connect(self.load_feature_cols)
        btn_feat_info = QPushButton("ℹ️")
        btn_feat_info.setFixedSize(24, 24)
        btn_feat_info.clicked.connect(lambda: self._show_info(self.tr("btn_load_features"), self.tr("help_feature_cols")))
        self.lbl_features = QLabel(self.tr("lbl_not_loaded"))
        f1.addWidget(self.btn_load_features)
        f1.addWidget(btn_feat_info)
        f1.addWidget(self.lbl_features)
        tabular_files_layout.addLayout(f1)

        f2 = QHBoxLayout()
        self.btn_load_label_enc = QPushButton(self.tr("btn_load_label_enc"))
        self.btn_load_label_enc.clicked.connect(self.load_label_encoder)
        btn_label_info = QPushButton("ℹ️")
        btn_label_info.setFixedSize(24, 24)
        btn_label_info.clicked.connect(lambda: self._show_info(self.tr("btn_load_label_enc"), self.tr("help_label_encoder")))
        self.lbl_label_enc = QLabel(self.tr("lbl_not_loaded"))
        f2.addWidget(self.btn_load_label_enc)
        f2.addWidget(btn_label_info)
        f2.addWidget(self.lbl_label_enc)
        tabular_files_layout.addLayout(f2)

        f3 = QHBoxLayout()
        self.btn_load_encoder = QPushButton(self.tr("btn_load_encoder"))
        self.btn_load_encoder.clicked.connect(self.load_onehot_encoder)
        btn_enc_info = QPushButton("ℹ️")
        btn_enc_info.setFixedSize(24, 24)
        btn_enc_info.clicked.connect(lambda: self._show_info(self.tr("btn_load_encoder"), self.tr("help_onehot_encoder")))
        self.lbl_encoder = QLabel(self.tr("lbl_not_loaded"))
        f3.addWidget(self.btn_load_encoder)
        f3.addWidget(btn_enc_info)
        f3.addWidget(self.lbl_encoder)
        tabular_files_layout.addLayout(f3)

        load_layout.addWidget(self.tabular_files_frame)
        layout.addWidget(load_frame)

        # Predict area
        pred_frame = QWidget()
        pred_layout = QVBoxLayout(pred_frame)
        pred_layout.setSpacing(6)
        pred_layout.addWidget(QLabel(self.tr("header_predict")))

        self.text_input = QTextEdit()
        self.text_input.setFixedHeight(120)
        self.file_input_frame = QWidget()
        file_layout = QHBoxLayout(self.file_input_frame)
        self.btn_select_file = QPushButton(self.tr("btn_file_select"))
        self.btn_select_file.clicked.connect(self.select_input_file)
        self.lbl_input_file = QLabel(self.tr("lbl_input_file"))
        file_layout.addWidget(self.btn_select_file)
        file_layout.addWidget(self.lbl_input_file)
        file_layout.addStretch(1)

        self.tabular_input_scroll = QScrollArea()
        self.tabular_input_scroll.setWidgetResizable(True)
        self.tabular_input_container = QWidget()
        self.tabular_input_layout = QFormLayout(self.tabular_input_container)
        self.tabular_input_scroll.setWidget(self.tabular_input_container)

        self.batch_frame = QWidget()
        batch_layout = QHBoxLayout(self.batch_frame)
        self.btn_select_csv = QPushButton(self.tr("btn_select_csv_batch"))
        self.btn_select_csv.clicked.connect(self.select_batch_csv)
        btn_batch_info = QPushButton("ℹ️")
        btn_batch_info.setFixedSize(24, 24)
        btn_batch_info.clicked.connect(lambda: self._show_info(self.tr("btn_select_csv_batch"), self.tr("help_batch_csv")))
        self.lbl_batch_csv = QLabel(self.tr("lbl_no_file"))
        self.btn_run_batch = QPushButton(self.tr("btn_run_batch"))
        self.btn_run_batch.clicked.connect(self.run_batch_prediction)
        batch_layout.addWidget(self.btn_select_csv)
        batch_layout.addWidget(btn_batch_info)
        batch_layout.addWidget(self.lbl_batch_csv)
        batch_layout.addWidget(self.btn_run_batch)
        batch_layout.addStretch(1)

        pred_layout.addWidget(self.text_input)
        pred_layout.addWidget(self.file_input_frame)
        pred_layout.addWidget(self.tabular_input_scroll)
        pred_layout.addWidget(self.batch_frame)

        self.btn_predict = QPushButton(self.tr("btn_predict"))
        self.btn_predict.clicked.connect(self.predict)
        pred_layout.addWidget(self.btn_predict)

        self.lbl_result = QLabel(self.tr("lbl_result_wait"))
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_layout.addWidget(self.lbl_result)

        layout.addWidget(pred_frame)

        self.on_type_change(self.tr("type_text"))

    def _show_info(self, title: str, text: str):
        content = f"<b>{title}</b><br/>{text}"
        QToolTip.showText(QCursor.pos(), content, self)

    def on_type_change(self, choice):
        self.model_type = self.type_map.get(choice, "text")

        self.text_input.setVisible(False)
        self.file_input_frame.setVisible(False)
        self.tabular_input_scroll.setVisible(False)
        self.tabular_files_frame.setVisible(False)
        self.batch_frame.setVisible(False)
        self.btn_load_extra.setEnabled(True)

        if self.model_type == "text":
            self.text_input.setVisible(True)
            self.btn_load_extra.setText(self.tr("btn_load_vectorizer"))
            self.lbl_extra_path.setText(self.tr("msg_load_vectorizer"))
        elif self.model_type in ["image", "audio"]:
            self.file_input_frame.setVisible(True)
            self.btn_load_extra.setText(self.tr("btn_load_scaler"))
            self.lbl_extra_path.setText(self.tr("lbl_scaler_optional"))
        elif self.model_type == "tabular":
            self.tabular_input_scroll.setVisible(True)
            self.tabular_files_frame.setVisible(True)
            self.batch_frame.setVisible(True)
            self.btn_load_extra.setEnabled(False)
            self.btn_load_extra.setText(self.tr("btn_load_extra"))
            self.lbl_extra_path.setText(self.tr("lbl_not_needed"))

    def load_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_select_model"), "", "Joblib Files (*.joblib)")
        if path:
            self.lbl_model_path.setText(os.path.basename(path))
            try:
                self.model = joblib.load(path)
                self._try_load_auxiliary_files(os.path.dirname(path))
                QMessageBox.information(self, self.tr("msg_warning"), self.tr("msg_model_loaded"))
            except Exception as e:
                QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_model_load_error").format(error=e))

    def _try_load_auxiliary_files(self, folder_path):
        if self.model_type == "text":
            vec_path = os.path.join(folder_path, "vectorizer.joblib")
            if os.path.exists(vec_path):
                try:
                    self.vectorizer = joblib.load(vec_path)
                    self.lbl_extra_path.setText(f"vectorizer.joblib {self.tr('lbl_loaded_auto')}")
                except Exception:
                    pass

        if self.model_type in ["image", "audio"]:
            scaler_path = os.path.join(folder_path, "scaler.joblib")
            if os.path.exists(scaler_path):
                try:
                    self.scaler = joblib.load(scaler_path)
                    self.lbl_extra_path.setText(f"scaler.joblib {self.tr('lbl_loaded_auto')}")
                except Exception:
                    pass

        if self.model_type == "tabular":
            fc_path = os.path.join(folder_path, "feature_cols.joblib")
            if os.path.exists(fc_path):
                try:
                    self.feature_cols = joblib.load(fc_path)
                    self.create_tabular_form()
                    self.lbl_features.setText(f"feature_cols.joblib {self.tr('lbl_loaded_auto')}")
                except Exception:
                    pass

            enc_path = os.path.join(folder_path, "encoder.joblib")
            if os.path.exists(enc_path):
                try:
                    self.encoder = joblib.load(enc_path)
                    self.lbl_encoder.setText(f"encoder.joblib {self.tr('lbl_loaded_auto')}")
                except Exception:
                    pass

            le_path = os.path.join(folder_path, "label_encoder.joblib")
            if os.path.exists(le_path):
                try:
                    self.label_encoder = joblib.load(le_path)
                    self.lbl_label_enc.setText(f"label_encoder.joblib {self.tr('lbl_loaded_auto')}")
                except Exception:
                    pass

    def load_extra_file(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_load_extra"), "", "Joblib Files (*.joblib)")
        if path:
            self.lbl_extra_path.setText(os.path.basename(path))
            try:
                obj = joblib.load(path)
                if self.model_type == "text":
                    self.vectorizer = obj
                else:
                    self.scaler = obj
                QMessageBox.information(self, self.tr("msg_warning"), self.tr("msg_file_loaded"))
            except Exception as e:
                QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_file_load_error").format(error=e))

    def load_feature_cols(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_load_features"), "", "Joblib Files (*.joblib)")
        if path:
            try:
                self.feature_cols = joblib.load(path)
                self.create_tabular_form()
                self.lbl_features.setText(os.path.basename(path))
                QMessageBox.information(self, self.tr("msg_warning"), self.tr("msg_file_loaded"))
            except Exception as e:
                QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_load_error").format(error=e))

    def load_label_encoder(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_load_label_enc"), "", "Joblib Files (*.joblib)")
        if path:
            try:
                self.label_encoder = joblib.load(path)
                self.lbl_label_enc.setText(os.path.basename(path))
                QMessageBox.information(self, self.tr("msg_warning"), self.tr("msg_file_loaded"))
            except Exception as e:
                QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_load_error").format(error=e))

    def load_onehot_encoder(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_load_encoder"), "", "Joblib Files (*.joblib)")
        if path:
            try:
                self.encoder = joblib.load(path)
                self.lbl_encoder.setText(os.path.basename(path))
                QMessageBox.information(self, self.tr("msg_warning"), self.tr("msg_file_loaded"))
            except Exception as e:
                QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_load_error").format(error=e))

    def create_tabular_form(self):
        while self.tabular_input_layout.rowCount():
            self.tabular_input_layout.removeRow(0)

        self.tabular_entries = {}
        if not self.feature_cols:
            return

        for col in self.feature_cols:
            entry = QLineEdit()
            self.tabular_input_layout.addRow(QLabel(col), entry)
            self.tabular_entries[col] = entry

    def select_input_file(self):
        if self.model_type == "image":
            filetypes = "Image Files (*.jpg *.png *.jpeg)"
        else:
            filetypes = "Audio Files (*.wav *.mp3 *.flac)"
        path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_file_select"), "", filetypes)
        if path:
            self.lbl_input_file.setText(os.path.basename(path))
            self.input_file_path = path

    def select_batch_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_select_csv_batch"), "", "CSV Files (*.csv)")
        if path:
            self.batch_csv_path = path
            self.lbl_batch_csv.setText(os.path.basename(path))

    def run_batch_prediction(self):
        if not self.model:
            QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_load_model_first"))
            return
        if not self.batch_csv_path:
            QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_select_csv_first"))
            return
        if not self.feature_cols:
            QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_load_features"))
            return

        try:
            df = pd.read_csv(self.batch_csv_path)
            missing_cols = [c for c in self.feature_cols if c not in df.columns]
            if missing_cols:
                QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_missing_columns").format(cols=", ".join(missing_cols)))
                return

            X = df[self.feature_cols].copy()
            for col in X.columns:
                try:
                    X[col] = pd.to_numeric(X[col])
                except (ValueError, TypeError):
                    pass

            if self.encoder:
                X_input = self.encoder.transform(X)
            else:
                X_input = X.values

            preds = self.model.predict(X_input)
            if self.label_encoder:
                try:
                    preds = self.label_encoder.inverse_transform(preds.astype(int))
                except Exception:
                    pass

            df_out = df.copy()
            df_out["prediction"] = preds

            save_path, _ = QFileDialog.getSaveFileName(self, self.tr("msg_save_predictions"), "", "CSV Files (*.csv)")
            if save_path:
                df_out.to_csv(save_path, index=False)
                QMessageBox.information(self, self.tr("msg_warning"), self.tr("msg_saved_predictions").format(path=os.path.basename(save_path)))
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_batch_predict_error").format(error=e))

    def predict(self):
        if not self.model:
            QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_load_model_first"))
            return

        input_data = None
        try:
            if self.model_type == "text":
                text = self.text_input.toPlainText().strip()
                if not text:
                    QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_enter_text"))
                    return
                if not self.vectorizer:
                    QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_load_vectorizer"))
                    return
                input_data = self.vectorizer.transform([text]).toarray()

            elif self.model_type == "image":
                if not self.input_file_path:
                    QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_select_image"))
                    return
                input_data = load_single_image(self.input_file_path)
                if self.scaler:
                    input_data = self.scaler.transform(input_data)

            elif self.model_type == "audio":
                if not self.input_file_path:
                    QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_select_audio"))
                    return
                input_data = load_single_audio(self.input_file_path)
                if self.scaler:
                    input_data = self.scaler.transform(input_data)

            elif self.model_type == "tabular":
                if not self.feature_cols:
                    QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_load_features"))
                    return

                data = {}
                for col, entry in self.tabular_entries.items():
                    data[col] = [entry.text()]
                df = pd.DataFrame(data)

                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except (ValueError, TypeError):
                        pass

                if self.encoder:
                    try:
                        input_data = self.encoder.transform(df)
                    except Exception as e:
                        QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_encoding_error").format(error=e))
                        return
                else:
                    input_data = df.values

            if input_data is None:
                raise ValueError(f"Input data not created. model_type={self.model_type}")

            prediction = self.model.predict(input_data)
            result_text = prediction[0]
            if self.label_encoder:
                try:
                    result_text = self.label_encoder.inverse_transform([int(result_text)])[0]
                except Exception:
                    pass

            confidence_text = ""
            if hasattr(self.model, "predict_proba"):
                try:
                    probs = self.model.predict_proba(input_data)
                    max_prob = np.max(probs)
                    confidence_text = self.tr("lbl_confidence").format(value=f"{max_prob * 100:.2f}")
                except Exception:
                    pass

            self.lbl_result.setText(self.tr("lbl_result").format(result=result_text) + confidence_text)
        except Exception as e:
            shape_info = getattr(input_data, "shape", "N/A")
            QMessageBox.critical(self, self.tr("msg_error"), f"{self.tr('msg_error')}: {e}\nInput Shape: {shape_info}")
