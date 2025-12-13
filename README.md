# Training Interface

**Training Interface**, makine öğrenimi modellerini kod yazmadan eğitmek, test etmek ve kullanmak için geliştirilmiş kapsamlı bir masaüstü uygulamasıdır. Metin, görüntü, ses ve tablosal veriler üzerinde işlem yapabilen modüler bir yapıya sahiptir.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Desteklenen Modeller](#-desteklenen-modeller)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Ekran Görüntüleri](#-ekran-görüntüleri)

## 🚀 Özellikler

Bu proje, farklı veri tipleri için özelleştirilmiş eğitim ve çıkarım (inference) modülleri sunar:

### 1. 📝 Metin Eğitimi (Text Training)
- Metin verileri üzerinde sınıflandırma modelleri eğitir.
- **TF-IDF** vektörleştirme yöntemini kullanır.
- Veri setlerini yükleyip eğitim/test olarak ayırabilir.

### 2. 🖼️ Görüntü İşleme (Image Training)
- Klasör tabanlı görüntü veri setlerini yükler (`root/class_name/image.jpg`).
- Görüntüleri otomatik olarak gri tonlamaya çevirir ve yeniden boyutlandırır (Varsayılan: 64x64).
- Piksel yoğunluklarını özellik olarak kullanır.

### 3. 🎵 Ses İşleme (Audio Training)
- Ses dosyalarını (`.wav`, `.mp3`, `.flac`) işler.
- **MFCC (Mel-frequency cepstral coefficients)** özellik çıkarımı yapar.
- Otomatik örnekleme oranı (sample rate) dönüşümü (16kHz) sağlar.

### 4. 📊 Tablosal Veri (Tabular Training)
- CSV formatındaki yapısal verileri destekler.
- Kategorik verileri otomatik olarak işler.
- Hedef değişken (target) seçimi ile esnek eğitim imkanı sunar.

### 5. 🧠 Çıkarım Modülü (Inference)
- Eğitilen modelleri (`.joblib` formatında) yükleyerek yeni veriler üzerinde tahmin yapmanızı sağlar.
- Tekil metin, görüntü veya ses dosyası yükleyerek anlık sonuç alabilirsiniz.

### 6. 🌐 Çoklu Dil Desteği
- Arayüz **Türkçe (TR)** ve **İngilizce (EN)** dillerini destekler.

## 🤖 Desteklenen Modeller

Uygulama, `scikit-learn` kütüphanesi tabanlı aşağıdaki algoritmaları destekler:

- **Naive Bayes** (Multinomial & Gaussian)
- **Support Vector Machines (SVM)** (Linear, RBF)
- **Random Forest**
- **Logistic Regression**
- **Decision Tree** (Gini & Entropy)
- **Gradient Boosting**
- **K-Nearest Neighbors (KNN)**

*Ayrıca Grid Search ile hiperparametre optimizasyonu seçeneği de mevcuttur.*

## ▶️ Kullanım

Uygulamayı başlatmak için ana dizindeki `main.py` dosyasını çalıştırın:

```bash
python main.py
```

Açılan arayüzde sol menüden çalışmak istediğiniz veri tipini seçerek işlemlere başlayabilirsiniz.

## 📂 Proje Yapısı

```
training-interface/
├── main.py                 # Uygulamanın giriş noktası
├── requirements.txt        # Bağımlılıklar
├── modules/                # Arka plan işlemleri
│   ├── data_loader.py      # Veri yükleme ve işleme (Görüntü, Ses, Metin)
│   ├── model_trainer.py    # Model tanımları ve eğitim fonksiyonları
│   ├── training_manager.py # Eğitim döngüsü yönetimi
│   ├── visualization.py    # Grafik çizdirme araçları
│   └── languages.py        # Dil dosyası
└── ui/                     # Kullanıcı Arayüzü (CustomTkinter)
    ├── main_window.py      # Ana pencere ve navigasyon
    ├── base_tab.py         # Ortak tab yapısı
    ├── text_tab.py         # Metin eğitimi arayüzü
    ├── image_tab.py        # Görüntü eğitimi arayüzü
    ├── audio_tab.py        # Ses eğitimi arayüzü
    ├── tabular_tab.py      # Tablosal veri eğitimi arayüzü
    └── inference_tab.py    # Tahminleme arayüzü
```

## 📸 Ekran Görüntüleri
<img width="1184" height="810" alt="last1" src="https://github.com/user-attachments/assets/6a8b4ad7-861f-42f9-82da-eeb7586736b4" />
<img width="428" height="550" alt="last2" src="https://github.com/user-attachments/assets/b5ec373d-7cb4-4317-9b17-e2688e755e94" />
<img width="1646" height="962" alt="resim" src="https://github.com/user-attachments/assets/5a199e49-14b1-437d-8fe6-37280698bea1" />



