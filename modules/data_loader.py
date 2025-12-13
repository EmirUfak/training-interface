import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchaudio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_images_from_folder(folder_path, img_size=(64, 64)):
    """
    Belirtilen klasörden görüntüleri yükler, gri tonlamaya çevirir ve düzleştirir.
    Klasör yapısı: root/class_name/image.jpg
    """
    data = []
    labels = []
    classes = os.listdir(folder_path)
    
    for category in classes:
        path = os.path.join(folder_path, category)
        if not os.path.isdir(path): continue
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path).convert('L')
                img = img.resize(img_size)
                img_array = np.array(img).flatten()
                data.append(img_array)
                labels.append(category)
            except Exception as e:
                print(f"Hata (Görüntü atlandı): {img_name} - {e}")
                pass
    
    return np.array(data), np.array(labels)

def load_audio_from_folder(folder_path, sample_rate=16000):
    data = []
    labels = []
    classes = os.listdir(folder_path)
    
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )

    for category in classes:
        path = os.path.join(folder_path, category)
        if not os.path.isdir(path): continue
        
        for audio_name in os.listdir(path):
            if not audio_name.lower().endswith(('.wav', '.mp3', '.flac')): continue
            try:
                audio_path = os.path.join(path, audio_name)
                waveform, sr = torchaudio.load(audio_path)
                
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                    waveform = resampler(waveform)

                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                mfcc = mfcc_transform(waveform)
                mfcc_mean = torch.mean(mfcc, dim=2).squeeze().numpy()
                
                data.append(mfcc_mean)
                labels.append(category)
            except Exception as e:
                print(f"Hata (Ses dosyası atlandı): {audio_name} - {e}")
                pass
                
    return np.array(data), np.array(labels)

def load_single_image(file_path, img_size=(64, 64)):
    try:
        img = Image.open(file_path).convert('L')
        img = img.resize(img_size)
        img_array = np.array(img).flatten()
        return img_array.reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Görüntü işlenemedi: {e}")

def load_single_audio(file_path, sample_rate=16000):
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

def load_and_vectorize_text(csv_path, text_col, label_col, max_features=2000, ngram_range=(1, 1), stop_words=None):
    df = pd.read_csv(csv_path)
    
    # Eğer text_col bir liste ise birleştir
    if isinstance(text_col, list):
        X = df[text_col].fillna('').astype(str).agg(' '.join, axis=1)
    else:
        X = df[text_col].astype(str)
        
    y = df[label_col]

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
    X_vec = vectorizer.fit_transform(X).toarray()
    
    return X_vec, y, vectorizer

def load_categorical_data(csv_path, feature_cols, label_col):
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
    
    # Hedef (y) için Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Preprocessor nesnesini de döndürüyoruz ki tahmin (inference) sırasında kullanabilelim
    return X_processed, y_encoded, preprocessor, label_encoder

