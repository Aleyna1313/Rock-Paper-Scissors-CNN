import torch
import torch.nn as nn

# PyTorch'ta her model nn.Module sınıfından miras alınarak oluşturulur
class GestureCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(GestureCNN, self).__init__()
        
        # 1. Konvolüsyon Bloğu (Özellik Çıkarımı)
        # Giriş: 3 kanal (RGB), Çıkış: 16 farklı özellik haritası
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64'ü 32x32'ye düşürür
        
        # 2. Konvolüsyon Bloğu
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32'yi 16x16'ya düşürür
        
        # 3. Konvolüsyon Bloğu
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16'yı 8x8'e düşürür
        
        # 3 boyutlu matrisi düzleştirici
        self.flatten = nn.Flatten()
        
        # Sınıflandırma (Karar) Bloğu
        # 64 özellik haritası * 8 yükseklik * 8 genişlik = 4096 adet girdi
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes) # Son çıkış: 3 sınıf
        
    def forward(self, x):
        # Verinin (x) katmanlardan sırasıyla geçiş (ileri yönlü yayılım) akışı
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.flatten(x) # Düzleştir
        
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x) # Son 3 skor
        
        return x

# Kodu test etmek için:
if __name__ == "__main__":
    model = GestureCNN(num_classes=3)
    
    # Rastgele bir veri oluştur (Batch=1, Kanal=3, Boyut=64x64)
    dummy_input = torch.randn(1, 3, 64, 64)
    
    # Modeli test et
    output = model(dummy_input)
    
    print(f"Modelin Çıktı Boyutu: {output.shape}")
    print("Model başarıyla tanımlandı!")