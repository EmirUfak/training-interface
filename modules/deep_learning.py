"""
Deep Learning (CNN) Modülü
Training Interface v2.0.0

PyTorch tabanlı CNN modelleri ile görüntü sınıflandırma desteği.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# GPU kullanılabilirlik kontrolü
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Deep Learning Device: {DEVICE}")


class ImageFolderDataset(Dataset):
    """
    Klasör tabanlı görüntü veri seti.
    Yapı: root/class_name/image.jpg
    """
    
    def __init__(self, image_paths, labels, img_size=(64, 64), transform=None):
        """
        Args:
            image_paths: Görüntü dosya yolları listesi
            labels: Etiketler (encoded)
            img_size: Hedef boyut (H, W)
            transform: Opsiyonel transform fonksiyonu
        """
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Görüntüyü yükle
        img = Image.open(img_path).convert('L')  # Grayscale
        img = img.resize(self.img_size)
        
        # Numpy -> Tensor
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # (1, H, W)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, torch.tensor(label, dtype=torch.long)


class SimpleCNN(nn.Module):
    """
    Basit CNN modeli - Görüntü sınıflandırma için.
    Input: (batch, 1, 64, 64) grayscale görüntüler
    """
    
    def __init__(self, num_classes, img_size=64):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )
        
        # Flatten size hesapla
        flatten_size = 128 * (img_size // 8) * (img_size // 8)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DeepCNN(nn.Module):
    """
    Daha derin CNN modeli - Zorlu veri setleri için.
    """
    
    def __init__(self, num_classes, img_size=64):
        super(DeepCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        flatten_size = 128 * (img_size // 8) * (img_size // 8)
        
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Model sözlüğü
CNN_MODELS = {
    "Simple CNN": SimpleCNN,
    "Deep CNN": DeepCNN,
}


def get_cnn_model(name: str, num_classes: int, img_size: int = 64) -> nn.Module:
    """
    CNN modeli döndürür.
    
    Args:
        name: Model ismi ("Simple CNN" veya "Deep CNN")
        num_classes: Sınıf sayısı
        img_size: Görüntü boyutu
    
    Returns:
        PyTorch model
    """
    if name not in CNN_MODELS:
        raise ValueError(f"Bilinmeyen model: {name}. Seçenekler: {list(CNN_MODELS.keys())}")
    
    model = CNN_MODELS[name](num_classes=num_classes, img_size=img_size)
    return model.to(DEVICE)


class CNNTrainer:
    """
    CNN eğitim yöneticisi.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        log_callback=None,
        progress_callback=None,
        stop_check=None
    ):
        """
        Args:
            model: PyTorch modeli
            learning_rate: Öğrenme oranı
            log_callback: Log mesajları için callback (msg, color)
            progress_callback: İlerleme için callback (epoch, total_epochs, loss, acc)
            stop_check: Durdurma kontrolü için callback
        """
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        self.log_callback = log_callback
        self.progress_callback = progress_callback
        self.stop_check = stop_check
        
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    def _log(self, msg, color="white"):
        if self.log_callback:
            self.log_callback(msg, color)
        logger.info(msg)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10
    ) -> dict:
        """
        Modeli eğitir.
        
        Args:
            train_loader: Eğitim veri yükleyici
            val_loader: Doğrulama veri yükleyici
            epochs: Epoch sayısı
        
        Returns:
            Eğitim geçmişi
        """
        self._log(f"🚀 CNN Eğitimi başlıyor ({epochs} epoch, device={DEVICE})", "cyan")
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            if self.stop_check and self.stop_check():
                self._log("🛑 Eğitim durduruldu.", "red")
                break
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                if self.stop_check and self.stop_check():
                    break
                
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            # Validation
            val_loss, val_acc = self._validate(val_loader)
            
            # Scheduler update
            self.scheduler.step(val_loss)
            
            # History
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            # Log
            self._log(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%",
                "yellow" if val_acc > best_val_acc else "white"
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if self.progress_callback:
                self.progress_callback(epoch + 1, epochs, val_loss, val_acc)
        
        self._log(f"✅ Eğitim tamamlandı. En iyi Val Acc: {best_val_acc:.2f}%", "green")
        return self.history
    
    def _validate(self, val_loader: DataLoader) -> tuple:
        """Validation step."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        return val_loss, val_acc
    
    def predict(self, data_loader: DataLoader) -> tuple:
        """
        Tahmin yapar.
        
        Returns:
            (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(DEVICE)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)


def prepare_cnn_data(
    folder_path: str,
    img_size: int = 64,
    test_size: float = 0.2,
    batch_size: int = 32
) -> tuple:
    """
    CNN eğitimi için veri hazırlar.
    
    Args:
        folder_path: Görüntü klasörü
        img_size: Hedef boyut
        test_size: Test oranı
        batch_size: Batch boyutu
    
    Returns:
        (train_loader, val_loader, label_encoder, num_classes)
    """
    # Dosyaları topla
    image_paths = []
    labels = []
    
    classes = sorted([d for d in os.listdir(folder_path) 
                     if os.path.isdir(os.path.join(folder_path, d))])
    
    for class_name in classes:
        class_path = os.path.join(folder_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(class_path, img_name))
                labels.append(class_name)
    
    # Label encoding
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    
    logger.info(f"Toplam {len(image_paths)} görüntü, {num_classes} sınıf bulundu.")
    
    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels_encoded, test_size=test_size, 
        stratify=labels_encoded, random_state=42
    )
    
    # Dataset oluştur
    train_dataset = ImageFolderDataset(X_train, y_train, img_size=(img_size, img_size))
    val_dataset = ImageFolderDataset(X_val, y_val, img_size=(img_size, img_size))
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, le, num_classes


def save_cnn_model(model: nn.Module, path: str, label_encoder=None, export_onnx: bool = True):
    """CNN modelini kaydeder."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }, path)
    
    if label_encoder:
        import joblib
        le_path = path.replace('.pt', '_label_encoder.joblib')
        joblib.dump(label_encoder, le_path)

    if export_onnx:
        try:
            dummy_input = torch.randn(1, 1, 64, 64).to(DEVICE)
            onnx_path = path.replace('.pt', '.onnx')
            torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"], opset_version=11)
            logger.info(f"ONNX export: {onnx_path}")
        except Exception as e:
            logger.warning(f"ONNX export atlandı: {e}")
    
    logger.info(f"Model kaydedildi: {path}")


def load_cnn_model(path: str, num_classes: int, img_size: int = 64) -> nn.Module:
    """CNN modelini yükler."""
    checkpoint = torch.load(path, map_location=DEVICE)
    model_class = checkpoint.get('model_class', 'SimpleCNN')
    
    if model_class == 'DeepCNN':
        model = DeepCNN(num_classes=num_classes, img_size=img_size)
    else:
        model = SimpleCNN(num_classes=num_classes, img_size=img_size)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    logger.info(f"Model yüklendi: {path}")
    return model
