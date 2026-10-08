import torch
import torch.nn as nn
import torch.optim as optim
import os

# Kendi yazdığımız dosyalardan model ve dataset'i çekiyoruz
from model import GestureCNN
from dataset import get_dataloaders

# --- AYARLAR ---
EPOCHS = 10               # Tüm veri setinin üzerinden kaç kez geçeceğiz?
LEARNING_RATE = 0.001     # Modelin öğrenme adımı (Çok büyük olursa hedefi ıskalar, çok küçük olursa hiç öğrenemez)
MODEL_SAVE_PATH = "../models/gesture_model.pth"

# Eğer bilgisayarında NVIDIA ekran kartı varsa GPU (cuda) kullan, yoksa işlemci (cpu) kullan
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim şu cihazda yapılıyor: {device}")

def train():
    # 1. Dataset ve Modeli Yükle
    train_loader, val_loader, _, classes = get_dataloaders()
    model = GestureCNN(num_classes=len(classes)).to(device)
    
    # 2. Hata (Loss) Fonksiyonu ve Optimizer (Güncelleyici)
    # CrossEntropyLoss: Sınıflandırma problemleri için standarttır (Softmax fonksiyonunu içinde barındırır)
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer: Geri yayılımda ağırlıkları güncelleyen en popüler ve akıllı algoritma
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # En iyi doğrulama başarısını takip etmek için
    best_val_acc = 0.0
    
    # Epoch Döngüsü (Tüm veriyi baştan sona kaç kez göreceği)
    for epoch in range(EPOCHS):
        
        # --- EĞİTİM FAZI ---
        model.train() # Modeli eğitim moduna al
        running_loss = 0.0
        
        for images, labels in train_loader:
            # Verileri (fotoğrafları ve etiketleri) CPU'dan GPU'ya (varsa) taşı
            images, labels = images.to(device), labels.to(device)
            
            # Adım 1: Sıfırlama
            optimizer.zero_grad() # Eski ağırlık güncellemelerini (gradyanları) temizle
            
            # Adım 2: Forward Pass (Tahmin Et)
            outputs = model(images)
            
            # Adım 3: Hata Hesapla (Loss)
            loss = criterion(outputs, labels)
            
            # Adım 4: Backward Pass (Geriye Dönük Türev Al)
            loss.backward()
            
            # Adım 5: Ağırlıkları Güncelle (Öğren)
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # --- DOĞRULAMA (VALIDATION) FAZI ---
        # Eğitimden sonra model ezberliyor mu yoksa gerçekten öğreniyor mu diye test ediyoruz
        model.eval() # Modeli test moduna al (Eğitim sırasındaki bazı rastgelelikleri kapatır)
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Test ederken modelin bir şeyler öğrenmesini (ağırlık güncellemesini) istemiyoruz.
        # torch.no_grad() hafıza kullanımını çok ciddi azaltır ve işlemi hızlandırır.
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # En yüksek skorlu sınıfı seç (örn: [0.1, 0.8, 0.1] -> index 1 kazanır)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: %{val_acc:.2f}")
        
        # Eğer bu epoch'taki başarı şu ana kadarki en iyisiyse, modeli kaydet!
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Klasör yoksa oluştur
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            # AŞAMA 8: Modeli Kaydetme
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   -> Yeni en iyi model kaydedildi! (Accuracy: %{val_acc:.2f})")

if __name__ == "__main__":
    train()