# Training Interface — Analiz Raporu (16 Ocak 2026)

## Kapsam
İncelenen dosyalar: [documents/AGENT.md](documents/AGENT.md), [README.md](README.md), [main.py](main.py), [modules/model_trainer.py](modules/model_trainer.py), [modules/training_manager.py](modules/training_manager.py), [modules/data_loader.py](modules/data_loader.py), [modules/deep_learning.py](modules/deep_learning.py), [modules/visualization.py](modules/visualization.py), [modules/languages.py](modules/languages.py), [ui/main_window.py](ui/main_window.py), [ui/base_tab.py](ui/base_tab.py), [ui/text_tab.py](ui/text_tab.py), [ui/image_tab.py](ui/image_tab.py), [ui/audio_tab.py](ui/audio_tab.py), [ui/tabular_tab.py](ui/tabular_tab.py), [ui/inference_tab.py](ui/inference_tab.py), [ui/results_manager.py](ui/results_manager.py).

---

## 1) Kod Kalitesi

### Güçlü Yönler
- Modüler yapı net: veri yükleme, eğitim, görselleştirme ve UI katmanları ayrılmış.
- Eğitim akışları için ortak `BaseTrainingTab` yaklaşımı iyi bir temel.
- ML tarafında hem klasik modeller hem de CNN desteği bulunması ürün kapsamını güçlendiriyor.
- `data_loader` içinde paralel yükleme kullanımı (image/audio) iyi bir yaklaşım.

### Sorunlar / Riskler
- **Parametre geçişi hatası:** `ImageTrainingTab` içinde `run_training_loop(...)` çağrısına `batch_mode`, `lazy_loader`, `epochs` gibi parametreler gönderiliyor; fakat `BaseTrainingTab.run_training_loop` bu parametreleri kabul etmiyor. Bu çağrı **TypeError** üretir.
- **Stop flag tutarsızlığı:** `BaseTrainingTab.stop_training()` içinde `stop_training_flag` kullanılıyor. Ancak `ImageTrainingTab._train_cnn_models()` içinde `self._stop_flag` kontrol ediliyor; bu **tanımlı değil**. Durdurma mekanizması CNN için çalışmayabilir.
- **Regresyon sonuçları UI tarafında yanlış işleniyor:** `TrainingManager` regresyon için `r2/mse/mae` döndürüyor ama `ResultsManager.show_model_result()` her zaman sınıflandırma metriklerine (`f1`, `accuracy`) erişiyor. Regresyon eğitiminde **KeyError** ve yanlış raporlama oluşur.
- **Kaydetme hatası:** `TrainingManager` içinde `save_model(model, ...)` çağrısı, GridSearch sonrası `res["model"]` yerine orijinal `model` nesnesini kaydediyor. En iyi model kaydı **yanlış** olabilir.
- **Logging standardı tutarsız:** `print()` kullanımı (özellikle `TrainingManager` ve `model_trainer`) ile `logging` karışık. Üretim seviyesinde tutarlılık ve seviyelendirme (info/warn/error) zayıf.
- **Type hints ve docstring eksikliği:** AGENT.md bunu özellikle istiyor ama çoğu public fonksiyonda yok.
- **i18n tutarsızlığı:** UI metinlerinin bir kısmı `languages.py` dışından hardcoded (ör. `TextTrainingTab` içinde “Max Kelime (Features)” vb.). Dil değişiminde tutarsız görünümler olur.
- **Hata geri bildirimleri:** Bazı yerlerde hata mesajları kullanıcıya gösteriliyor ama log ayrıntıları kayboluyor. UI’da teknik hata ayrıntısı ve kullanıcı mesajı ayrıştırılmalı.

---

## 2) Performans

### Olumlu Noktalar
- Görsel ve ses yükleme paralelleştirilmiş.
- Tabular veri için `ColumnTransformer` + pipeline yaklaşımı iyi pratik.
- CNN eğitiminde `DataLoader` kullanımı uygun.

### Performans Riskleri
- **TF-IDF yoğun matris:** `load_and_vectorize_text(...).toarray()` çok büyük veri setlerinde RAM’i hızla doldurur. Sparse matrisi desteklemek daha uygun.
- **Image yükleme flatten + RAM:** Görsellerin tamamını `numpy` array’e çekmek büyük veri setinde ölçeklenmez. Batch/streaming yolu sadece image eğitiminde var, diğer veri tipleri için yok.
- **Audio MFCC hesaplama:** Her dosya için MFCC oluşturmak CPU yoğun. `max_workers` küçük ama yine de büyük setlerde uzun sürer. Caching veya precompute seçenekleri yok.
- **GridSearch default n_jobs=-1:** Küçük makinelerde UI donmasına yol açabilir. Arka plan thread olsa da CPU saturation hissedilir.
- **Sonuç görselleri ve CSV kaydı:** Her model için plot ve disk IO yapılıyor. Büyük model sayısında UI akıcılığı etkilenebilir.

---

## 3) Eksik Yönler ve Eklenebilecekler

### Fonksiyonel Eksikler
- **K-Fold / CV desteği:** AGENT.md’de kısa vadeli hedef ama UI’da yok.
- **Gelişmiş metin ön işleme:** Stop-word, lemmatization, dil tespiti, n-gram auto-tuning.
- **Model raporlarının dışa aktarımı:** Eğitim raporu (HTML/PDF) ve parametre özetleri yok.
- **Model versiyonlama / model kartı:** Kayıt altında dataset meta, parametreler, metrikler ve etiketlerin tutulması eksik.
- **Model açıklanabilirliği:** SHAP/LIME entegrasyonu yok.
- **Model export:** ONNX/TFLite ve Python kod export planlanmış ama UI/altyapı yok.
- **Inference iyileştirmeleri:** Batch inference, CSV üzerinden toplu tahmin, sonuçların export’u yok.

### Teknik Borç
- **Test yok:** Unit/integration test altyapısı görünmüyor.
- **Konfigürasyon yönetimi yok:** Varsayılanlar UI koduna gömülü.
- **Hata yakalama standardı:** Çok farklı yaklaşım var.
- **Tip doğrulama:** Kullanıcı girişleri için kapsamlı validasyon eksik.

---

## 4) UI Kullanılabilirliği ve Görsellik

### Mevcut Artılar
- Modül tabanlı, sol sidebar düzeni anlaşılır.
- Eğitim akışında sonuçların görselleştirilmesi değerli.
- Dark theme ve tutarlı UI bileşenleri (CustomTkinter) hoş.

### İyileştirme Önerileri

#### Kullanılabilirlik
- **Durum/Log alanı sabitlenmeli:** Eğitim sırasında loglar kayıyor; üstte sabit “Durum paneli” daha okunur olur.
- **Form validasyonları:** Giriş alanlarında anlık doğrulama ve tooltip önerisi.
- **Eğitim iptali:** Durdurma butonu tüm iş akışlarını tutarlı durdurmalı. CNN için stop flag tutarsızlığı giderilmeli.
- **Model seçimi:** “Tümü seç”/“Tümü temizle” hızlı kontroller eklenebilir.
- **Sonuç filtreleme:** Model sonuçlarında metrik bazlı sıralama/filtreleme.
- **Dosya yolları:** UI’da tam yol yerine kısaltılmış “...” gösterimi, hover tooltip ile tam yol.
- **Regresyon sonuçları:** Ayrı sonuç kartı ve grafikler (residual, predicted vs actual) UI’ya bağlanmalı.

#### Görsellik
- **Renk hiyerarşisi:** Başlıklar ve metrik kartları için tutarlı renk paleti.
- **Kart tasarımı:** Model sonuçları kartlarında ikon + metriğe göre renklendirme.
- **Grafik boyutları:** Confusion Matrix ve feature importance grafiklerinin ölçeği, ekran boyutuna göre dinamik ayarlanmalı.
- **Spacing:** Tab içindeki komponentler arasında daha belirgin boşluk/hiyerarşi.

#### i18n
- UI üzerindeki hardcoded tüm metinler `languages.py` içine taşınmalı.
- `EN/TR` dil değişiminde state korunmalı (ör. seçili dosya adı, model seçimleri).

---

## Özet ve Öncelikli Aksiyon Listesi

### Kritik (Hemen)
1. `ImageTrainingTab` → `BaseTrainingTab.run_training_loop` parametre uyumsuzluğu düzeltilmeli.
2. `self._stop_flag` yerine tek bir durdurma flag’i kullanılmalı.
3. Regresyon sonuçlarının UI’da doğru render edilmesi sağlanmalı.
4. GridSearch sonrası doğru modelin kaydedilmesi garantilenmeli.

### Orta Öncelik
1. Logging standardizasyonu (`logging` + seviyeler).
2. i18n metinlerinin tamamının `languages.py` içine alınması.
3. TF-IDF ve veri yükleme süreçlerinde bellek dostu seçenekler.

### Uzun Vadeli
1. Export/AutoML özellikleri ve model kartları.
2. Batch inference ve rapor/export seçenekleri.
3. UI/UX polish (tema, ikonlar, layout düzeni).
