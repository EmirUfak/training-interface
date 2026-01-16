import pandas as pd
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtWidgets import QToolTip
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QComboBox, QSpinBox, QMessageBox, QTableView, QHeaderView, QCheckBox,
    QGroupBox, QGridLayout, QSizePolicy
)

from modules.languages import get_text
import modules.data_prep as data_prep


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self._df.iat[index.row(), index.column()]
        return "" if pd.isna(value) else str(value)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


class DatasetEditorTab(QWidget):
    def __init__(self, lang="tr"):
        super().__init__()
        self.lang = lang
        self.df = pd.DataFrame()
        self.preview_df = pd.DataFrame()
        self.raw_df = pd.DataFrame()

        self._build_ui()

    def tr(self, key):
        return get_text(key, self.lang)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        file_row = QHBoxLayout()
        btn_load = QPushButton(self.tr("btn_file_select"))
        btn_load.clicked.connect(self.load_csv)
        btn_load.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.lbl_path = QLabel(self.tr("lbl_no_file"))
        file_row.addWidget(btn_load)
        file_row.addWidget(self.lbl_path)
        file_row.addStretch(1)
        layout.addLayout(file_row)

        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel(self.tr("lbl_preview_rows")))
        self.spin_rows = QSpinBox()
        self.spin_rows.setMinimum(5)
        self.spin_rows.setMaximum(200)
        self.spin_rows.setValue(10)
        self.spin_rows.valueChanged.connect(self.refresh_preview)
        self.spin_rows.setMaximumWidth(100)
        preview_row.addWidget(self.spin_rows)
        preview_row.addStretch(1)
        layout.addLayout(preview_row)

        self.table = QTableView()
        self.model = DataFrameModel(self.preview_df)
        self.table.setModel(self.model)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setMaximumSectionSize(220)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.table)

        self.combo_col = QComboBox()
        self.combo_fill = QComboBox()
        self.combo_fill.addItems(["median", "mean", "mode"])
        self.chk_lower = QCheckBox(self.tr("lbl_lowercase"))
        self.chk_lower.setChecked(True)
        self.chk_punct = QCheckBox(self.tr("lbl_remove_punct"))
        self.chk_punct.setChecked(True)

        ops_box = QGroupBox(self.tr("lbl_ops"))
        ops_grid = QGridLayout(ops_box)
        ops_grid.setAlignment(Qt.AlignmentFlag.AlignLeft)
        ops_grid.setSizeConstraint(QGridLayout.SizeConstraint.SetFixedSize)
        ops_info = QPushButton("ℹ️")
        ops_info.setFixedSize(28, 28)
        ops_info.setStyleSheet("padding: 4px;")
        ops_info.clicked.connect(lambda: self._show_info(self.tr("lbl_ops"), self.tr("info_ops")))
        ops_grid.addWidget(ops_info, 0, 4)
        ops_grid.addWidget(QLabel(self.tr("lbl_column")), 0, 0)
        ops_grid.addWidget(self.combo_col, 0, 1)
        ops_grid.addWidget(QLabel(self.tr("lbl_fill_strategy")), 0, 2)
        ops_grid.addWidget(self.combo_fill, 0, 3)
        self.combo_col.setMaximumWidth(200)
        self.combo_fill.setMaximumWidth(160)
        ops_grid.addWidget(self.chk_lower, 1, 0)
        ops_grid.addWidget(self.chk_punct, 1, 1)

        self.btn_drop_rows = QPushButton(self.tr("btn_drop_rows"))
        self.btn_drop_rows.clicked.connect(self.drop_selected_rows)
        self.btn_drop_cols = QPushButton(self.tr("btn_drop_cols"))
        self.btn_drop_cols.clicked.connect(self.drop_selected_cols)
        self.btn_dedup = QPushButton(self.tr("btn_dedup"))
        self.btn_dedup.clicked.connect(self.deduplicate)
        self.btn_fill = QPushButton(self.tr("btn_fill_missing"))
        self.btn_fill.clicked.connect(self.fill_missing)
        self.btn_clean = QPushButton(self.tr("btn_clean_text"))
        self.btn_clean.clicked.connect(self.clean_text)
        self.btn_tokenize = QPushButton(self.tr("btn_tokenize"))
        self.btn_tokenize.clicked.connect(self.tokenize_text)
        for b in [self.btn_drop_rows, self.btn_drop_cols, self.btn_dedup, self.btn_fill, self.btn_clean, self.btn_tokenize]:
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        ops_grid.addWidget(self.btn_drop_rows, 2, 0)
        ops_grid.addWidget(self.btn_drop_cols, 2, 1)
        ops_grid.addWidget(self.btn_dedup, 2, 2)
        ops_grid.addWidget(self.btn_fill, 2, 3)
        ops_grid.addWidget(self.btn_clean, 3, 0)
        ops_grid.addWidget(self.btn_tokenize, 3, 1)
        layout.addWidget(ops_box)

        label_box = QGroupBox(self.tr("lbl_label_ops"))
        label_grid = QGridLayout(label_box)
        label_grid.setAlignment(Qt.AlignmentFlag.AlignLeft)
        label_grid.setSizeConstraint(QGridLayout.SizeConstraint.SetFixedSize)
        label_info = QPushButton("ℹ️")
        label_info.setFixedSize(28, 28)
        label_info.setStyleSheet("padding: 4px;")
        label_info.clicked.connect(lambda: self._show_info(self.tr("lbl_label_ops"), self.tr("info_label_ops")))
        label_grid.addWidget(label_info, 0, 4)
        label_grid.addWidget(QLabel(self.tr("lbl_text_column")), 0, 0)
        self.combo_text_col = QComboBox()
        label_grid.addWidget(self.combo_text_col, 0, 1)
        label_grid.addWidget(QLabel(self.tr("lbl_label_column")), 0, 2)
        self.combo_label_col = QComboBox()
        label_grid.addWidget(self.combo_label_col, 0, 3)
        self.combo_text_col.setMaximumWidth(200)
        self.combo_label_col.setMaximumWidth(200)
        self.btn_export = QPushButton(self.tr("btn_export_labeling"))
        self.btn_export.clicked.connect(self.export_for_labeling)
        self.btn_import = QPushButton(self.tr("btn_import_labels"))
        self.btn_import.clicked.connect(self.import_labeled)
        self.btn_export.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.btn_import.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        label_grid.addWidget(self.btn_export, 1, 0)
        label_grid.addWidget(self.btn_import, 1, 1)
        layout.addWidget(label_box)

        filter_box = QGroupBox(self.tr("lbl_filter_ops"))
        filter_grid = QGridLayout(filter_box)
        filter_grid.setAlignment(Qt.AlignmentFlag.AlignLeft)
        filter_grid.setSizeConstraint(QGridLayout.SizeConstraint.SetFixedSize)
        filter_info = QPushButton("ℹ️")
        filter_info.setFixedSize(28, 28)
        filter_info.setStyleSheet("padding: 4px;")
        filter_info.clicked.connect(lambda: self._show_info(self.tr("lbl_filter_ops"), self.tr("info_filter_ops")))
        filter_grid.addWidget(filter_info, 0, 2)
        filter_grid.addWidget(QLabel(self.tr("lbl_filter_text")), 0, 0)
        self.txt_filter = QComboBox()
        self.txt_filter.setEditable(True)
        filter_grid.addWidget(self.txt_filter, 0, 1)
        self.txt_filter.setMaximumWidth(220)
        self.btn_filter = QPushButton(self.tr("btn_apply_filter"))
        self.btn_filter.clicked.connect(self.apply_filter)
        self.btn_reset = QPushButton(self.tr("btn_reset_filter"))
        self.btn_reset.clicked.connect(self.reset_filter)
        self.btn_append = QPushButton(self.tr("btn_append_csv"))
        self.btn_append.clicked.connect(self.append_csv)
        self.btn_filter.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.btn_reset.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.btn_append.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        filter_grid.addWidget(self.btn_filter, 1, 0)
        filter_grid.addWidget(self.btn_reset, 1, 1)
        filter_grid.addWidget(self.btn_append, 1, 2)
        layout.addWidget(filter_box)

        footer_row = QHBoxLayout()
        self.btn_save = QPushButton(self.tr("btn_save_clean"))
        self.btn_save.clicked.connect(self.save_csv)
        self.status_lbl = QLabel(self.tr("msg_ready"))
        self.btn_save.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.status_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        footer_row.addWidget(self.btn_save)
        footer_row.addWidget(self.status_lbl)
        footer_row.addStretch(1)
        layout.addLayout(footer_row)

        self._apply_tooltips()

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_file_select"), "", "CSV Files (*.csv)")
        if not file_path:
            return
        try:
            self.df = pd.read_csv(file_path)
            self.raw_df = self.df.copy()
            self.lbl_path.setText(file_path)
            self.combo_col.clear()
            self.combo_col.addItems(self.df.columns.tolist())
            self.combo_text_col.clear()
            self.combo_label_col.clear()
            self.combo_text_col.addItems(self.df.columns.tolist())
            self.combo_label_col.addItems(self.df.columns.tolist())
            self.txt_filter.clear()
            self.refresh_preview()
            self._set_status(self.tr("msg_loaded").format(count=len(self.df)))
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_file_read_error").format(error=e))

    def refresh_preview(self):
        if self.df.empty:
            return
        n = int(self.spin_rows.value())
        self.preview_df = self.df.head(n).copy()
        self.model.set_dataframe(self.preview_df)
        self._update_table_height()

    def _selected_row_indices(self):
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedIndexes()})
        return rows

    def _selected_col_indices(self):
        cols = sorted({idx.column() for idx in self.table.selectionModel().selectedIndexes()})
        return cols

    def drop_selected_rows(self):
        if self.df.empty:
            return
        rows = self._selected_row_indices()
        if not rows:
            return
        index_values = self.preview_df.index[rows]
        self.df = data_prep.drop_rows(self.df, index_values)
        self.raw_df = self.df.copy()
        self.refresh_preview()
        self._set_status(self.tr("msg_rows_deleted").format(count=len(index_values)))

    def drop_selected_cols(self):
        if self.df.empty:
            return
        cols = self._selected_col_indices()
        if not cols:
            return
        col_names = [self.preview_df.columns[c] for c in cols]
        self.df = data_prep.drop_cols(self.df, col_names)
        self.raw_df = self.df.copy()
        self.combo_col.clear()
        self.combo_col.addItems(self.df.columns.tolist())
        self.refresh_preview()
        self._set_status(self.tr("msg_cols_deleted").format(count=len(col_names)))

    def deduplicate(self):
        if self.df.empty:
            return
        col = self.combo_col.currentText()
        subset = [col] if col else None
        self.df = data_prep.remove_duplicates(self.df, subset)
        self.raw_df = self.df.copy()
        self.refresh_preview()
        self._set_status(self.tr("msg_dedup_done"))

    def fill_missing(self):
        if self.df.empty:
            return
        strategy = self.combo_fill.currentText()
        col = self.combo_col.currentText()
        cols = [col] if col else None
        self.df = data_prep.fill_missing(self.df, strategy=strategy, cols=cols)
        self.raw_df = self.df.copy()
        self.refresh_preview()
        self._set_status(self.tr("msg_fill_done").format(strategy=strategy))

    def clean_text(self):
        if self.df.empty:
            return
        col = self.combo_col.currentText()
        if not col:
            return
        self.df = data_prep.clean_text(self.df, col, lower=self.chk_lower.isChecked(), remove_punct=self.chk_punct.isChecked())
        self.raw_df = self.df.copy()
        self.refresh_preview()
        self._set_status(self.tr("msg_clean_done"))

    def tokenize_text(self):
        if self.df.empty:
            return
        col = self.combo_col.currentText()
        if not col:
            return
        self.df = data_prep.tokenize_text(self.df, col, lowercase=self.chk_lower.isChecked())
        self.raw_df = self.df.copy()
        self.refresh_preview()
        self._set_status(self.tr("msg_tokenize_done"))

    def save_csv(self):
        if self.df.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, self.tr("btn_save_clean"), "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.df.to_csv(path, index=False)
            QMessageBox.information(self, self.tr("msg_warning"), self.tr("msg_saved").format(path=path))
            self._set_status(self.tr("msg_saved").format(path=path))
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_save_error").format(error=e))
            self._set_status(self.tr("msg_save_error").format(error=e))

    def export_for_labeling(self):
        if self.df.empty:
            return
        text_col = self.combo_text_col.currentText()
        label_col = self.combo_label_col.currentText()
        if not text_col:
            return
        export_df = pd.DataFrame({
            "id": range(len(self.df)),
            "text": self.df[text_col].astype(str)
        })
        if label_col:
            export_df["label"] = self.df[label_col] if label_col in self.df.columns else ""

        path, _ = QFileDialog.getSaveFileName(self, self.tr("btn_export_labeling"), "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            export_df.to_csv(path, index=False)
            QMessageBox.information(self, self.tr("msg_warning"), self.tr("msg_saved").format(path=path))
            self._set_status(self.tr("msg_export_done"))
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_save_error").format(error=e))
            self._set_status(self.tr("msg_save_error").format(error=e))

    def import_labeled(self):
        if self.df.empty:
            return
        path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_import_labels"), "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            labeled = pd.read_csv(path)
            if "id" not in labeled.columns or "label" not in labeled.columns:
                QMessageBox.warning(self, self.tr("msg_warning"), self.tr("msg_label_file_invalid"))
                return
            label_col = self.combo_label_col.currentText() or "label"
            self.df[label_col] = self.df.index.map(labeled.set_index("id")["label"])
            self.refresh_preview()
            QMessageBox.information(self, self.tr("msg_warning"), self.tr("msg_labels_imported"))
            self._set_status(self.tr("msg_labels_imported"))
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_save_error").format(error=e))
            self._set_status(self.tr("msg_save_error").format(error=e))

    def apply_filter(self):
        if self.df.empty:
            return
        col = self.combo_col.currentText()
        text = self.txt_filter.currentText().strip()
        if not col or not text:
            return
        self.df = self.df[self.df[col].astype(str).str.contains(text, case=False, na=False)]
        self.raw_df = self.df.copy()
        self.refresh_preview()
        self._set_status(self.tr("msg_filter_done"))

    def reset_filter(self):
        if self.raw_df.empty:
            return
        self.df = self.raw_df.copy()
        self.refresh_preview()
        self._set_status(self.tr("msg_filter_reset"))

    def append_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("btn_append_csv"), "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            new_df = pd.read_csv(path)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self.raw_df = self.df.copy()
            self.refresh_preview()
            self._set_status(self.tr("msg_append_done").format(count=len(new_df)))
        except Exception as e:
            QMessageBox.critical(self, self.tr("msg_error"), self.tr("msg_file_read_error").format(error=e))
            self._set_status(self.tr("msg_file_read_error").format(error=e))

    def _set_status(self, text: str):
        if hasattr(self, "status_lbl"):
            self.status_lbl.setText(text)

    def _show_info(self, title: str, text: str):
        content = f"<b>{title}</b><br/>{text}"
        QToolTip.showText(QCursor.pos(), content, self)

    def _apply_tooltips(self):
        self.btn_drop_rows.setToolTip(self.tr("tip_drop_rows"))
        self.btn_drop_cols.setToolTip(self.tr("tip_drop_cols"))
        self.btn_dedup.setToolTip(self.tr("tip_dedup"))
        self.btn_fill.setToolTip(self.tr("tip_fill"))
        self.btn_clean.setToolTip(self.tr("tip_clean"))
        self.btn_tokenize.setToolTip(self.tr("tip_tokenize"))
        self.btn_export.setToolTip(self.tr("tip_export_labels"))
        self.btn_import.setToolTip(self.tr("tip_import_labels"))
        self.btn_filter.setToolTip(self.tr("tip_filter"))
        self.btn_reset.setToolTip(self.tr("tip_filter_reset"))
        self.btn_append.setToolTip(self.tr("tip_append"))
        self.btn_save.setToolTip(self.tr("tip_save"))

    def _update_table_height(self):
        target = max(220, int(self.height() * 0.3))
        self.table.setFixedHeight(target)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_table_height()
