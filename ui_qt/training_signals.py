from PyQt6.QtCore import QObject, pyqtSignal


class TrainingSignals(QObject):
    log = pyqtSignal(str, str)
    result = pyqtSignal(str, dict, object, object, str, dict)
    comparison = pyqtSignal(list, str, dict)
    best_model = pyqtSignal(dict, str, dict)
    completion = pyqtSignal(str)
    error = pyqtSignal(str, Exception)
