from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QLabel, QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QToolTip

from modules.languages import get_text
from ui_qt.text_tab import TextTrainingTab
from ui_qt.image_tab import ImageTrainingTab
from ui_qt.audio_tab import AudioTrainingTab
from ui_qt.tabular_tab import TabularTrainingTab
from ui_qt.inference_tab import InferenceTab


class TrainingInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = "tr"
        self.setWindowTitle(get_text("app_title", self.current_lang))

        self.root = QWidget()
        self.setCentralWidget(self.root)
        self.main_layout = QHBoxLayout(self.root)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(8)

        self._apply_font()
        self._build_sidebar()
        self._build_stack()
        self._apply_window_geometry()

    def toggle_language(self):
        self.current_lang = "en" if self.current_lang == "tr" else "tr"
        self.setWindowTitle(get_text("app_title", self.current_lang))
        current_index = self.stack.currentIndex() if hasattr(self, "stack") else 0
        self._build_sidebar()
        self._build_stack()
        self.stack.setCurrentIndex(min(current_index, self.stack.count() - 1))

    def _apply_font(self):
        font = QFont()
        font.setPointSize(12)
        app = QApplication.instance()
        if app:
            app.setFont(font)
        QToolTip.setFont(QFont("Segoe UI", 10))
        if app:
            tooltip_style = (
                "QToolTip {"
                "background-color: #1f1f1f;"
                "color: #f5f5f5;"
                "border: 1px solid #3a3a3a;"
                "border-radius: 8px;"
                "padding: 8px;"
                "max-width: 220px;"
                "}"
            )
            app.setStyleSheet((app.styleSheet() or "") + tooltip_style)

    def _apply_window_geometry(self):
        screen = self.screen() or QApplication.primaryScreen()
        if not screen:
            self.resize(1000, 700)
            return
        geom = screen.availableGeometry()
        width = int(geom.width() * 0.7)
        height = int(geom.height() * 0.7)
        x = geom.x() + (geom.width() - width) // 2
        y = geom.y() + (geom.height() - height) // 2
        self.setGeometry(x, y, width, height)

    def _build_sidebar(self):
        if hasattr(self, "sidebar"):
            self.main_layout.removeWidget(self.sidebar)
            self.sidebar.deleteLater()

        self.sidebar = QWidget()
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        sidebar_layout.setSpacing(8)
        sidebar_layout.setContentsMargins(6, 6, 6, 6)

        self.title_label = QLabel(get_text("app_title", self.current_lang))
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        sidebar_layout.addWidget(self.title_label)

        self.btn_text = QPushButton(get_text("sidebar_text", self.current_lang))
        self.btn_image = QPushButton(get_text("sidebar_image", self.current_lang))
        self.btn_audio = QPushButton(get_text("sidebar_audio", self.current_lang))
        self.btn_tabular = QPushButton(get_text("sidebar_tabular", self.current_lang))
        self.btn_infer = QPushButton(get_text("sidebar_inference", self.current_lang))
        self.btn_lang = QPushButton(get_text("lang_toggle", self.current_lang))

        for b in [self.btn_text, self.btn_image, self.btn_audio, self.btn_tabular, self.btn_infer, self.btn_lang]:
            sidebar_layout.addWidget(b)

        self.main_layout.addWidget(self.sidebar, 0)

    def _build_stack(self):
        if hasattr(self, "stack"):
            self.main_layout.removeWidget(self.stack)
            self.stack.deleteLater()

        self.stack = QStackedWidget()
        self.text_tab = TextTrainingTab(lang=self.current_lang)
        self.image_tab = ImageTrainingTab(lang=self.current_lang)
        self.audio_tab = AudioTrainingTab(lang=self.current_lang)
        self.tabular_tab = TabularTrainingTab(lang=self.current_lang)
        self.inference_tab = InferenceTab(lang=self.current_lang)

        self.stack.addWidget(self.text_tab)
        self.stack.addWidget(self.image_tab)
        self.stack.addWidget(self.audio_tab)
        self.stack.addWidget(self.tabular_tab)
        self.stack.addWidget(self.inference_tab)

        self.main_layout.addWidget(self.stack, 1)

        self.btn_text.clicked.connect(lambda: self.stack.setCurrentWidget(self.text_tab))
        self.btn_image.clicked.connect(lambda: self.stack.setCurrentWidget(self.image_tab))
        self.btn_audio.clicked.connect(lambda: self.stack.setCurrentWidget(self.audio_tab))
        self.btn_tabular.clicked.connect(lambda: self.stack.setCurrentWidget(self.tabular_tab))
        self.btn_infer.clicked.connect(lambda: self.stack.setCurrentWidget(self.inference_tab))
        self.btn_lang.clicked.connect(self.toggle_language)
