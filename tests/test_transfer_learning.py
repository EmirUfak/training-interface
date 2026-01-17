import torch
import torch.nn as nn
from modules import transfer_learning


class DummyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)


class DummyMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(10, 2))


def test_build_transfer_model_monkeypatch(monkeypatch):
    def _resnet(pretrained=True):
        return DummyResNet()

    def _mobilenet(pretrained=True):
        return DummyMobileNet()

    monkeypatch.setattr(transfer_learning, "TL_MODELS", {
        "ResNet18": _resnet,
        "MobileNetV2": _mobilenet,
    })

    model = transfer_learning.build_transfer_model("ResNet18", num_classes=3, freeze_base=True)
    assert isinstance(model, nn.Module)

    model2 = transfer_learning.build_transfer_model("MobileNetV2", num_classes=3, freeze_base=False)
    assert isinstance(model2, nn.Module)
