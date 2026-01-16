"""
Data Augmentation Modülü
Training Interface v2.1.0

Görüntü veri setlerini çoğaltmak için augmentation teknikleri sağlar.
"""

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import logging

logger = logging.getLogger(__name__)


class ImageAugmentor:
    """
    Görüntü veri çoğaltma (augmentation) sınıfı.
    
    Desteklenen transformasyonlar:
    - Döndürme (rotation)
    - Yatay/Dikey çevirme (flip)
    - Parlaklık değişimi (brightness)
    - Kontrast değişimi (contrast)
    - Gaussian gürültü (noise)
    - Bulanıklaştırma (blur)
    """
    
    def __init__(
        self,
        rotation_range: float = 30.0,
        flip_horizontal: bool = True,
        flip_vertical: bool = False,
        brightness_range: tuple = (0.7, 1.3),
        contrast_range: tuple = (0.8, 1.2),
        noise_factor: float = 0.05,
        blur_probability: float = 0.2
    ):
        """
        Args:
            rotation_range: Maksimum döndürme açısı (derece)
            flip_horizontal: Yatay çevirme aktif mi
            flip_vertical: Dikey çevirme aktif mi
            brightness_range: Parlaklık aralığı (min, max)
            contrast_range: Kontrast aralığı (min, max)
            noise_factor: Gürültü miktarı (0-1)
            blur_probability: Bulanıklaştırma olasılığı (0-1)
        """
        self.rotation_range = rotation_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_factor = noise_factor
        self.blur_probability = blur_probability
    
    def augment(self, img: Image.Image) -> Image.Image:
        """
        Tek bir görüntüye rastgele augmentation uygular.
        
        Args:
            img: PIL Image nesnesi
            
        Returns:
            Augmented PIL Image
        """
        # 1. Döndürme
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            img = img.rotate(angle, fillcolor=0, expand=False)
        
        # 2. Yatay çevirme
        if self.flip_horizontal and random.random() > 0.5:
            img = ImageOps.mirror(img)
        
        # 3. Dikey çevirme
        if self.flip_vertical and random.random() > 0.5:
            img = ImageOps.flip(img)
        
        # 4. Parlaklık
        if self.brightness_range != (1.0, 1.0):
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        
        # 5. Kontrast
        if self.contrast_range != (1.0, 1.0):
            factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        
        # 6. Bulanıklaştırma
        if random.random() < self.blur_probability:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        return img
    
    def add_noise(self, img_array: np.ndarray) -> np.ndarray:
        """Numpy array'e Gaussian gürültü ekler."""
        if self.noise_factor <= 0:
            return img_array
        
        noise = np.random.normal(0, self.noise_factor * 255, img_array.shape)
        noisy = img_array + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def augment_dataset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        multiplier: int = 3,
        img_size: tuple = (64, 64)
    ) -> tuple:
        """
        Veri setini augmentation ile çoğaltır.
        
        Args:
            images: Orijinal görüntü arrayi (N, H*W) veya (N, H, W)
            labels: Etiketler
            multiplier: Çoğaltma katsayısı (her örnek için kaç tane üretilecek)
            img_size: Görüntü boyutu (yükseklik, genişlik)
            
        Returns:
            (augmented_images, augmented_labels) tuple'ı
        """
        augmented_data = []
        augmented_labels = []
        
        total = len(images)
        logger.info(f"Data Augmentation başlatıldı: {total} örnek x {multiplier} = {total * multiplier}")
        
        for i, (img_array, label) in enumerate(zip(images, labels)):
            # Orijinali ekle
            augmented_data.append(img_array)
            augmented_labels.append(label)
            
            # Görüntüyü PIL formatına çevir
            try:
                if img_array.ndim == 1:
                    # Flatten edilmiş (H*W,) -> (H, W)
                    img_2d = img_array.reshape(img_size)
                else:
                    img_2d = img_array
                
                img = Image.fromarray(img_2d.astype(np.uint8), mode='L')
                
                # Augmented versiyonları ekle
                for _ in range(multiplier - 1):
                    aug_img = self.augment(img)
                    aug_array = np.array(aug_img)
                    
                    # Gürültü ekle
                    if self.noise_factor > 0:
                        aug_array = self.add_noise(aug_array)
                    
                    augmented_data.append(aug_array.flatten())
                    augmented_labels.append(label)
                    
            except Exception as e:
                logger.warning(f"Augmentation hatası (index {i}): {e}")
                continue
        
        logger.info(f"Augmentation tamamlandı: {len(augmented_data)} toplam örnek")
        return np.array(augmented_data), np.array(augmented_labels)


# Ön tanımlı augmentation profilleri
AUGMENTATION_PROFILES = {
    "light": ImageAugmentor(
        rotation_range=15,
        flip_horizontal=True,
        brightness_range=(0.9, 1.1),
        noise_factor=0.02
    ),
    "medium": ImageAugmentor(
        rotation_range=30,
        flip_horizontal=True,
        flip_vertical=False,
        brightness_range=(0.7, 1.3),
        contrast_range=(0.8, 1.2),
        noise_factor=0.05
    ),
    "heavy": ImageAugmentor(
        rotation_range=45,
        flip_horizontal=True,
        flip_vertical=True,
        brightness_range=(0.5, 1.5),
        contrast_range=(0.6, 1.4),
        noise_factor=0.1,
        blur_probability=0.3
    )
}


def get_augmentor(profile: str = "medium") -> ImageAugmentor:
    """
    Ön tanımlı augmentation profili döndürür.
    
    Args:
        profile: "light", "medium", veya "heavy"
        
    Returns:
        ImageAugmentor instance
    """
    return AUGMENTATION_PROFILES.get(profile, AUGMENTATION_PROFILES["medium"])
