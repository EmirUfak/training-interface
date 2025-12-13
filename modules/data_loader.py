import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchaudio
from sklearn.feature_extraction.text import TfidfVectorizer

def load_images_from_folder(folder_path, img_size=(64, 64)):
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
            except Exception:
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
                print(f"Error processing {audio_name}: {e}")
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

def load_and_vectorize_text(csv_path, text_col, label_col, max_features=2000):
    df = pd.read_csv(csv_path)
    X = df[text_col].astype(str)
    y = df[label_col]

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_vec = vectorizer.fit_transform(X).toarray()
    
    return X_vec, y, vectorizer
