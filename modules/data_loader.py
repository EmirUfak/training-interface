import os
import re
import logging
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchaudio
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Logger ayarla
logger = logging.getLogger(__name__)

_TR_STOP_WORDS = {
    "ve", "veya", "ama", "fakat", "çünkü", "için", "ile", "de", "da", "bu", "şu", "o",
    "bir", "iki", "çok", "az", "daha", "en", "gibi", "ne", "niye", "nasıl", "mı", "mi",
    "mü", "mu", "ben", "sen", "biz", "siz", "onlar", "her", "hiç", "var", "yok",
}


def get_stop_words(language: str):
    if language == "turkish":
        return list(_TR_STOP_WORDS)
    if language == "english":
        return "english"
    return None

def _load_single_image(args):
    """Tek bir görüntüyü yükleyen yardımcı fonksiyon (paralel işlem için)."""
    img_path, img_size, category = args
    try:
        img = Image.open(img_path).convert('L')
        img = img.resize(img_size)
        img_array = np.array(img).flatten()
        return (img_array, category, None)
    except Exception as e:
        return (None, None, f"Hata (Görüntü atlandı): {os.path.basename(img_path)} - {e}")

def load_images_from_folder(folder_path: str, img_size=(64, 64), max_workers=None):
    """
    Belirtilen klasörden görüntüleri paralel olarak yükler, gri tonlamaya çevirir ve düzleştirir.
    Klasör yapısı: root/class_name/image.jpg
    
    Args:
        folder_path: Veri seti klasörü
        img_size: Hedef görüntü boyutu (default: 64x64)
        max_workers: Paralel işlem sayısı (default: CPU çekirdek sayısı)
    """
    # Tüm dosya yollarını topla
    image_tasks = []
    classes = os.listdir(folder_path)
    
    for category in classes:
        path = os.path.join(folder_path, category)
        if not os.path.isdir(path): 
            continue
        
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(path, img_name)
                image_tasks.append((img_path, img_size, category))
    
    data = []
    labels = []
    
    # Paralel yükleme
    workers = max_workers or min(os.cpu_count() or 4, 8)
    logger.info(f"Görüntüler {workers} worker ile paralel yükleniyor ({len(image_tasks)} dosya)...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_load_single_image, image_tasks)
        
        for img_array, category, error in results:
            if error:
                logger.warning(error)
            elif img_array is not None:
                data.append(img_array)
                labels.append(category)
    
    logger.info(f"Toplam {len(data)} görüntü başarıyla yüklendi.")
    return np.array(data), np.array(labels)

def _load_single_audio(args):
    """Tek bir ses dosyasını yükleyen yardımcı fonksiyon (paralel işlem için)."""
    audio_path, sample_rate, mfcc_transform, category, cache_dir = args
    try:
        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_name = os.path.basename(audio_path) + f"_{sample_rate}.npy"
            cache_path = os.path.join(cache_dir, cache_name)
            if os.path.exists(cache_path):
                return (np.load(cache_path), category, None)

        waveform, sr = torchaudio.load(audio_path)
        
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mfcc = mfcc_transform(waveform)
        mfcc_mean = torch.mean(mfcc, dim=2).squeeze().numpy()

        if cache_path:
            np.save(cache_path, mfcc_mean)
        
        return (mfcc_mean, category, None)
    except Exception as e:
        return (None, None, f"Hata (Ses dosyası atlandı): {os.path.basename(audio_path)} - {e}")

def load_audio_from_folder(folder_path: str, sample_rate: int = 16000, max_workers=None, use_cache: bool = True):
    """
    Belirtilen klasörden ses dosyalarını paralel olarak yükler ve MFCC çıkarır.
    
    Args:
        folder_path: Veri seti klasörü
        sample_rate: Hedef örnekleme oranı (default: 16000)
        max_workers: Paralel işlem sayısı (default: CPU çekirdek sayısı / 2)
    """
    # MFCC transform'u bir kez oluştur (tüm dosyalar için yeniden kullanılır)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    
    # Tüm dosya yollarını topla
    audio_tasks = []
    classes = os.listdir(folder_path)
    cache_dir = os.path.join(folder_path, ".cache") if use_cache else None
    
    for category in classes:
        path = os.path.join(folder_path, category)
        if not os.path.isdir(path): 
            continue
        
        for audio_name in os.listdir(path):
            if audio_name.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                audio_path = os.path.join(path, audio_name)
                audio_tasks.append((audio_path, sample_rate, mfcc_transform, category, cache_dir))
    
    data = []
    labels = []
    
    # Paralel yükleme (ses işleme IO-bound olduğu için thread kullanıyoruz)
    workers = max_workers or min((os.cpu_count() or 4) // 2, 4)
    logger.info(f"Ses dosyaları {workers} worker ile paralel yükleniyor ({len(audio_tasks)} dosya)...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_load_single_audio, audio_tasks)
        
        for mfcc_mean, category, error in results:
            if error:
                logger.warning(error)
            elif mfcc_mean is not None:
                data.append(mfcc_mean)
                labels.append(category)
    
    logger.info(f"Toplam {len(data)} ses dosyası başarıyla yüklendi.")
    return np.array(data), np.array(labels)

def load_single_image(file_path: str, img_size=(64, 64)):
    try:
        img = Image.open(file_path).convert('L')
        img = img.resize(img_size)
        img_array = np.array(img).flatten()
        return img_array.reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Görüntü işlenemedi: {e}")

def load_single_audio(file_path: str, sample_rate: int = 16000):
    try:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
        )
        
        waveform, sr = torchaudio.load(file_path)
        
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mfcc = mfcc_transform(waveform)
        mfcc_mean = torch.mean(mfcc, dim=2).squeeze().numpy()
        
        return mfcc_mean.reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Ses dosyası işlenemedi: {e}")

def _basic_text_preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_vectorize_text(
    csv_path,
    text_col,
    label_col,
    max_features=2000,
    ngram_range=(1, 1),
    stop_words=None,
    use_sparse: bool = True,
    preprocess_text: bool = True,
):
    """Metin verisini yükler ve TF-IDF vektörizer ile dönüştürür."""
    df = pd.read_csv(csv_path)
    
    # Eğer text_col bir liste ise birleştir
    if isinstance(text_col, list):
        X = df[text_col].fillna('').astype(str).agg(' '.join, axis=1)
    else:
        X = df[text_col].astype(str)
        
    y = df[label_col]

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
        preprocessor=_basic_text_preprocess if preprocess_text else None,
    )
    X_vec = vectorizer.fit_transform(X)

    if not use_sparse:
        X_vec = X_vec.toarray()
    
    return X_vec, y, vectorizer

def load_categorical_data(csv_path: str, feature_cols, label_col, is_regression: bool = False):
    """
    Tablo verilerini (Sayısal + Kategorik) işlemek için gelişmiş fonksiyon.
    Otomatik olarak sayısal ve kategorik sütunları ayırır, eksik verileri doldurur ve ölçeklendirir.
    """
    df = pd.read_csv(csv_path)
    
    X = df[feature_cols]
    y = df[label_col]

    # Sütun tiplerini otomatik algıla
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    # 1. Sayısal Veriler İçin Pipeline (Eksik Veri Doldurma + Ölçeklendirme)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Eksik verileri ortalama ile doldur
        ('scaler', StandardScaler())                 # Verileri normalize et
    ])

    # 2. Kategorik Veriler İçin Pipeline (Eksik Veri Doldurma + One-Hot Encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Eksik verileri en sık geçenle doldur
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # İşlemleri Birleştir
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Seçilmeyen sütunları at
    )

    # Dönüşümü Uygula
    X_processed = preprocessor.fit_transform(X)
    
    # Hedef (y) için İşleme
    label_encoder = None
    y_encoded = None

    if is_regression:
        # Regresyon ise sayısal olmalı, encoder yok
        # Eğer y string ise float'a çevirmeyi dene, başarısız olursa hata ver
        try:
             y_encoded = y.astype(float)
        except ValueError:
             raise ValueError("Regresyon hedefi sayısal olmalıdır.")
    else:
        # Sınıflandırma ise Label Encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

    # Preprocessor nesnesini de döndürüyoruz ki tahmin (inference) sırasında kullanabilelim
    return X_processed, y_encoded, preprocessor, label_encoder


class LazyImageLoader:
    def __init__(self, folder_path, img_size=(64, 64)):
        self.folder_path = folder_path
        self.img_size = img_size
        self.classes = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
        self.files = []
        self.labels = []
        
        # Dosya yollarını tara
        for label in self.classes:
            class_path = os.path.join(folder_path, label)
            for f in os.listdir(class_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.files.append(os.path.join(class_path, f))
                    self.labels.append(label)
        
        # Label Encoding manuel yapılıyor çünkü partial_fit sayısal değer ister
        from sklearn.preprocessing import LabelEncoder
        self.le = LabelEncoder()
        self.le.fit(self.classes)
        self.y_encoded = self.le.transform(self.labels)

    def get_split(self, test_size=0.2):
        """
        Dosya yollarını train/test olarak ayırır.
        """
        X_train_files, X_test_files, y_train, y_test = train_test_split(
            self.files, self.y_encoded, test_size=test_size, stratify=self.y_encoded, random_state=42
        )
        return X_train_files, X_test_files, y_train, y_test

    def yield_batch(self, file_paths, labels, batch_size=32, shuffle=True):
        """
        Generator: Belirtilen dosya listesinden parça parça görüntü okur ve döndürür.
        """
        n_samples = len(file_paths)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            
            batch_X = []
            batch_y = []
            
            for i in batch_idx:
                try:
                    img = Image.open(file_paths[i]).convert('L')
                    img = img.resize(self.img_size)
                    img_array = np.array(img).flatten()
                    batch_X.append(img_array)
                    batch_y.append(labels[i])
                except Exception as e:
                    logger.warning(f"Hata (LazyLoader): {file_paths[i]} okunamadı - {e}")
            
            if len(batch_X) > 0:
                yield np.array(batch_X), np.array(batch_y)
