import logging
import sys
from PyQt6.QtWidgets import QApplication
from ui_qt.main_window import TrainingInterface

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        app = QApplication(sys.argv)
        window = TrainingInterface()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.getLogger(__name__).exception("Uygulama hatası")
