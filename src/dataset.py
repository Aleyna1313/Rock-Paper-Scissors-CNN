import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Tüm fotoğrafları aynı boyuta getireceğiz
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32

# 1. Eğitim verisi için Transforms (Hafif Augmentasyon içerir)
train_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomRotation(degrees=10), # Sağ/sol 10 derece döndür
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Işıkla hafif oyna
    transforms.ToTensor(), # Pikselleri 0-1 arası tensöre çevir
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # -1 ile +1 arasına sıkıştır
])

# 2. Validation ve Test için Transforms (Asla augmentasyon yapılmaz, sadece netleştirme)
val_test_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_dataloaders(base_dir="../dataset_split"):
    # Klasör yolları
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")
    
    # ImageFolder ile klasör isimlerinden etiket (0,1,2) oluşturma
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)
    
    print(f"Sınıflar ve İndeksleri: {train_dataset.class_to_idx}")
    
    # DataLoader motorlarını oluşturma
    # Eğitim verisi modelin ezberlememesi için karıştırılır (shuffle=True)
    # Val ve Test verisi sadece değerlendirme içindir, karıştırmaya gerek yoktur.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.classes

# Kodu test etmek için
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    
    # Train loader'dan sadece 1 paket (batch) çekip boyutlarına bakalım
    images, labels = next(iter(train_loader))
    print(f"1 Paketin Görüntü Boyutu [Batch, Channel, Height, Width]: {images.shape}")
    print(f"1 Paketin Etiket Boyutu: {labels.shape}")