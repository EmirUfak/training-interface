# Training Interface

Bu proje, makine öğrenimi modellerini eğitmek ve görselleştirmek için geliştirilmiş bir kullanıcı arayüzüdür. Metin, görüntü ve ses verileri üzerinde işlem yapabilen modüller içerir.

## Özellikler

- **Metin Eğitimi:** Metin verileri üzerinde model eğitimi ve test işlemleri.
- **Görüntü İşleme:** Görüntü verileri için eğitim arayüzü.
- **Ses İşleme:** Ses verileri için eğitim arayüzü.
- **Görselleştirme:** Eğitim sonuçlarının grafiksel gösterimi.

## Kurulum

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:

```bash
pip install -r requirements.txt
```

## Kullanım

Uygulamayı başlatmak için `main.py` dosyasını çalıştırın:

```bash
python main.py
```

## Proje Yapısı

- `main.py`: Uygulamanın giriş noktası.
- `modules/`: Veri yükleme, model eğitimi ve görselleştirme modülleri.
- `ui/`: Kullanıcı arayüzü bileşenleri (sekmeler ve ana pencere).
