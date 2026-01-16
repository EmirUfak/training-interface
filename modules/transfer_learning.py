import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TL_MODELS = {
    "ResNet18": models.resnet18,
    "MobileNetV2": models.mobilenet_v2,
    "EfficientNet-B0": models.efficientnet_b0,
}


def build_transfer_model(name: str, num_classes: int, freeze_base: bool = True):
    if name not in TL_MODELS:
        raise ValueError(f"Unknown model: {name}")
    model = TL_MODELS[name](pretrained=True)

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    if name.startswith("ResNet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name.startswith("MobileNet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name.startswith("EfficientNet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model.to(DEVICE)


def train_transfer_learning(
    data_dir: str,
    model_name: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    freeze_base: bool = True,
    log_callback=None,
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(data_dir, transform=transform)
    num_classes = len(dataset.classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = build_transfer_model(model_name, num_classes=num_classes, freeze_base=freeze_base)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    def _log(msg):
        if log_callback:
            log_callback(msg, "cyan")
        logger.info(msg)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()
        acc = correct / max(1, total)
        _log(f"🧠 Transfer Learning epoch {epoch+1}/{epochs} - loss={running_loss:.4f} acc={acc:.4f}")

    return model, acc, dataset.classes


def save_transfer_model(model, path: str, class_names, model_name: str):
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "class_names": class_names,
    }, path)
