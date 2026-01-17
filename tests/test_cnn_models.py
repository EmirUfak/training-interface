import torch
from modules.deep_learning import get_cnn_model


def test_get_cnn_model_simple():
    model = get_cnn_model("Simple CNN", num_classes=3, img_size=64)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 3)


def test_get_cnn_model_deep():
    model = get_cnn_model("Deep CNN", num_classes=4, img_size=64)
    x = torch.randn(1, 1, 64, 64)
    y = model(x)
    assert y.shape == (1, 4)
