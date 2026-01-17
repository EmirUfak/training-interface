# Training Interface

**English version below → [English Version](#-english-version)**

**Training Interface**, makine öğrenimi modellerini kod yazmadan eğitmek, test etmek ve kullanmak için geliştirilmiş kapsamlı bir masaüstü uygulamasıdır. Metin, görüntü, ses ve tablosal veriler üzerinde işlem yapabilen modüler bir yapıya sahiptir.

**Training Interface** is a desktop app for training, testing, and using machine learning models without writing code. It supports text, image, audio, and tabular workflows with a modular UI.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Desteklenen Modeller](#-desteklenen-modeller)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Ekran Görüntüleri](#-ekran-görüntüleri)

## 🚀 Özellikler (Features)

Bu proje, farklı veri tipleri için özelleştirilmiş eğitim ve çıkarım (inference) modülleri sunar:

### 1. 📝 Metin Eğitimi (Text Training)
- Metin verileri üzerinde sınıflandırma modelleri eğitir.
- **TF-IDF** vektörleştirme yöntemini kullanır.
- Veri setlerini yükleyip eğitim/test olarak ayırabilir.
- Seyrek TF-IDF ve stop-words seçenekleri ile bellek/dil optimizasyonu.

### 2. 🖼️ Görüntü İşleme (Image Training)
- Klasör tabanlı görüntü veri setlerini yükler (`root/class_name/image.jpg`).
- Görüntüleri otomatik olarak gri tonlamaya çevirir ve yeniden boyutlandırır (Varsayılan: 64x64).
- Piksel yoğunluklarını özellik olarak kullanır.
- Düşük bellek modu (batch) ve veri çoğaltma (augmentation) desteği.

### 3. 🎵 Ses İşleme (Audio Training)
- Ses dosyalarını (`.wav`, `.mp3`, `.flac`) işler.
- **MFCC (Mel-frequency cepstral coefficients)** özellik çıkarımı yapar.
- Otomatik örnekleme oranı (sample rate) dönüşümü (16kHz) sağlar.

### 4. 📊 Tablosal Veri (Tabular Training)
- CSV formatındaki yapısal verileri destekler.
- Kategorik verileri otomatik olarak işler.
- Hedef değişken (target) seçimi ile esnek eğitim imkanı sunar.
- Sınıflandırma ve regresyon görevleri için model seçimi.

### 5. 🧠 Çıkarım Modülü (Inference)
- Eğitilen modelleri (`.joblib` formatında) yükleyerek yeni veriler üzerinde tahmin yapmanızı sağlar.
- Tekil metin, görüntü veya ses dosyası yükleyerek anlık sonuç alabilirsiniz.
- Tablosal veriler için CSV ile toplu tahmin ve dışa aktarım.

### 6. 🧹 Veri Düzenleme (Dataset Editor)
- CSV önizleme, satır/sütun silme, dedup, eksik doldurma ve metin temizleme.
- Etiketleme için dışa aktarım ve geri içe alma.

### 7. 🌐 Çoklu Dil Desteği (TR/EN)
- Arayüz **Türkçe (TR)** ve **İngilizce (EN)** dillerini destekler.

### 8. 📦 Çıktı Seçenekleri (Outputs)
- Eğitim çıktıları artık `results/` altında tarih damgalı klasörlerde saklanır.
- Model, veri setleri, vectorizer/scaler, grafikler, özet raporlar ve model kartları isteğe bağlı kaydedilir.

### 9. 🧩 Gelişmiş Öğrenme
- **Ensemble (Voting)** ve **ROC eğrisi** desteği.
- **Transfer Learning** (ResNet18 / MobileNetV2 / EfficientNet-B0).
- **Federated (deneysel)** simülasyonu (sınıflandırma).

## 🤖 Desteklenen Modeller (Supported Models)

Uygulama, `scikit-learn` kütüphanesi tabanlı aşağıdaki algoritmaları destekler:

- **Naive Bayes** (Multinomial & Gaussian)
- **Support Vector Machines (SVM/SVR)** (Linear, RBF, Poly, Sigmoid)
- **Random Forest** / **Random Forest Regressor**
- **Logistic Regression**
- **Decision Tree** (Gini & Entropy) / **Decision Tree Regressor**
- **Gradient Boosting** / **Gradient Boosting Regressor**
- **K-Nearest Neighbors (KNN/KNN Regressor)**
- **Linear Regression**, **Ridge**, **Lasso**
- **Simple CNN**, **Deep CNN** (image)

*Ayrıca Grid Search ile hiperparametre optimizasyonu seçeneği de mevcuttur.*

## ▶️ Kullanım (Usage)

Uygulamayı başlatmak için ana dizindeki `main.py` dosyasını çalıştırın:

```bash
python main.py
```

Açılan arayüzde sol menüden çalışmak istediğiniz veri tipini seçerek işlemlere başlayabilirsiniz.

The UI uses PyQt6. Run `main.py` to launch the app.

## 📂 Proje Yapısı (Project Structure)

```
training-interface/
├── main.py                 # Uygulamanın giriş noktası
├── requirements.txt        # Bağımlılıklar
├── modules/                # Arka plan işlemleri
│   ├── data_loader.py      # Veri yükleme ve işleme (Görüntü, Ses, Metin)
│   ├── data_prep.py        # Veri temizleme / düzenleme
│   ├── model_trainer.py    # Model tanımları ve eğitim fonksiyonları
│   ├── training_manager.py # Eğitim döngüsü yönetimi
│   ├── transfer_learning.py# Transfer learning yardımcıları
│   ├── federated.py        # Federated (simülasyon)
│   ├── visualization.py    # Grafik çizdirme araçları
│   └── languages.py        # Dil dosyası
├── ui_qt/                  # Kullanıcı Arayüzü (PyQt6)
│   ├── main_window.py      # Ana pencere ve navigasyon
│   ├── base_tab.py         # Ortak tab yapısı
│   ├── text_tab.py         # Metin eğitimi arayüzü
│   ├── image_tab.py        # Görüntü eğitimi arayüzü
│   ├── audio_tab.py        # Ses eğitimi arayüzü
│   ├── tabular_tab.py      # Tablosal veri eğitimi arayüzü
│   ├── inference_tab.py    # Tahminleme arayüzü
│   └── dataset_editor_tab.py # Veri düzenleme arayüzü
└── results/                # Eğitim çıktıları (tarih damgalı klasörler)
```

## 📸 Ekran Görüntüleri
<img width="1333" height="838" alt="ss1" src="https://github.com/user-attachments/assets/abe2a6de-f014-4f7f-a821-e5057ccfe51e" />
<img width="1335" height="843" alt="ss2" src="https://github.com/user-attachments/assets/b3ef527f-1cb0-43fe-b78b-a61ff166dc95" />

---

# English Version

**Screenshots are below → [Screenshots](#-screenshots)**

## 📋 Table of Contents

- [Features](#-features)
- [Supported Models](#-supported-models)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)

## 🚀 Features

Training Interface provides modular training and inference workflows for multiple data types:

### 1. 📝 Text Training
- Trains classification models on text data.
- Uses **TF-IDF** vectorization.
- Split datasets into train/test.
- Sparse TF-IDF and stop-words options for memory/language optimization.

### 2. 🖼️ Image Training
- Loads folder-based image datasets (`root/class_name/image.jpg`).
- Auto grayscale + resize (default: 64x64).
- Uses pixel intensities as features.
- Low-memory batch mode and augmentation support.

### 3. 🎵 Audio Training
- Processes `.wav`, `.mp3`, `.flac` audio.
- Extracts **MFCC** features.
- Auto resampling to 16kHz.

### 4. 📊 Tabular Training
- Supports CSV structured data.
- Handles categorical features automatically.
- Flexible target selection for classification/regression.

### 5. 🧠 Inference
- Load trained models (`.joblib`) and run predictions on new data.
- Single text/image/audio inference.
- Batch CSV inference for tabular data with export.

### 6. 🧹 Dataset Editor
- CSV preview, row/column delete, dedup, missing fill, text cleanup.
- Label export/import for annotation flows.

### 7. 🌐 Multi-language UI (TR/EN)
- Interface supports **Turkish (TR)** and **English (EN)**.

### 8. 📦 Outputs
- Training outputs are saved under `results/` with timestamped folders.
- Optional saving of models, datasets, vectorizer/scaler, plots, summary reports, and model cards.

### 9. 🧩 Advanced Learning
- **Ensemble (Voting)** and **ROC curve** support.
- **Transfer Learning** (ResNet18 / MobileNetV2 / EfficientNet-B0).
- **Federated (experimental)** simulation (classification).

## 🤖 Supported Models

The app supports the following algorithms via `scikit-learn`:

- **Naive Bayes** (Multinomial & Gaussian)
- **Support Vector Machines (SVM/SVR)** (Linear, RBF, Poly, Sigmoid)
- **Random Forest** / **Random Forest Regressor**
- **Logistic Regression**
- **Decision Tree** (Gini & Entropy) / **Decision Tree Regressor**
- **Gradient Boosting** / **Gradient Boosting Regressor**
- **K-Nearest Neighbors (KNN/KNN Regressor)**
- **Linear Regression**, **Ridge**, **Lasso**
- **Simple CNN**, **Deep CNN** (image)

*Hyperparameter optimization via Grid Search is also available.*

## ▶️ Usage

Run `main.py` from the project root:

```bash
python main.py
```

The UI uses PyQt6. Pick a data type from the left sidebar to begin.

## 📂 Project Structure

```
training-interface/
├── main.py                 # App entry point
├── requirements.txt        # Dependencies
├── modules/                # Backend operations
│   ├── data_loader.py      # Data loading (image/audio/text)
│   ├── data_prep.py        # Data cleaning / editing
│   ├── model_trainer.py    # Models and training
│   ├── training_manager.py # Training loop manager
│   ├── transfer_learning.py# Transfer learning helpers
│   ├── federated.py        # Federated (simulation)
│   ├── visualization.py    # Plotting utilities
│   └── languages.py        # UI language strings
├── ui_qt/                  # UI (PyQt6)
│   ├── main_window.py      # Main window + navigation
│   ├── base_tab.py         # Shared tab layout
│   ├── text_tab.py         # Text training UI
│   ├── image_tab.py        # Image training UI
│   ├── audio_tab.py        # Audio training UI
│   ├── tabular_tab.py      # Tabular training UI
│   ├── inference_tab.py    # Inference UI
│   └── dataset_editor_tab.py # Dataset editor UI
└── results/                # Training outputs (timestamped)
```

## 📸 Screenshots
<img width="1333" height="838" alt="ss1" src="https://github.com/user-attachments/assets/abe2a6de-f014-4f7f-a821-e5057ccfe51e" />
<img width="1335" height="843" alt="ss2" src="https://github.com/user-attachments/assets/b3ef527f-1cb0-43fe-b78b-a61ff166dc95" />

