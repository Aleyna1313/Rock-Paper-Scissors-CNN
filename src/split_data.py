import os
import shutil
import random

# Yollar (src klasöründen bir üst klasöre çıkıp dataset'i buluyoruz)
DATASET_DIR = "../dataset"
OUTPUT_DIR = "../dataset_split"
CLASSES = ["fist", "palm", "victory"]

# Bölme oranları
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

print("Veriler ayrıştırılıyor, lütfen bekleyin...")

# Önce yeni klasör yapısını oluşturalım
for split in SPLIT_RATIOS.keys():
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# Her sınıf için fotoğrafları okuyup bölelim
for cls in CLASSES:
    src_folder = os.path.join(DATASET_DIR, cls)
    # Sadece .jpg dosyalarını al
    images = [img for img in os.listdir(src_folder) if img.endswith(".jpg")]
    
    # Listeyi rastgele karıştır ki hep aynı açılar aynı klasöre düşmesin
    random.seed(42) # Her çalıştırmada aynı rastgeleliği versin diye sabitliyoruz
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])
    
    # Listeyi dilimliyoruz
    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]
    
    # Şimdi dosyaları ilgili klasörlere kopyalayalım
    for img in train_imgs:
        shutil.copy(os.path.join(src_folder, img), os.path.join(OUTPUT_DIR, "train", cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(src_folder, img), os.path.join(OUTPUT_DIR, "val", cls, img))
    for img in test_imgs:
        shutil.copy(os.path.join(src_folder, img), os.path.join(OUTPUT_DIR, "test", cls, img))
        
    print(f"{cls.upper()} Sınıfı -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

print("\nİşlem Tamamlandı! Yeni dataset_split klasörünüz hazır.")