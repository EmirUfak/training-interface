import numpy as np
from modules.visualization import create_roc_curve_figure


def test_roc_curve_figure():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    fig = create_roc_curve_figure(y_true, y_proba)
    assert fig is not None
