# Training Interface

Bu proje, makine öğrenimi modellerini eğitmek ve görselleştirmek için geliştirilmiş bir kullanıcı arayüzüdür. Metin, görüntü ve ses verileri üzerinde işlem yapabilen modüller içerir.

<img width="1897" height="1010" alt="interface-1" src="https://github.com/user-attachments/assets/ad6fd156-7c28-4b6d-aee1-6386c096b0a0" />
<img width="1890" height="978" alt="interface-2" src="https://github.com/user-attachments/assets/af903d0e-c9cb-4c40-861e-e6ba1331a9ed" />
<img width="1667" height="760" alt="interface-3" src="https://github.com/user-attachments/assets/6560c201-4de1-4cf3-91f2-b7eb080ed18a" />


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
