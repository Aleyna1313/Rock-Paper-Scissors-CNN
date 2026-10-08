import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import GestureCNN

# Ayarlar
MODEL_PATH = "../models/gesture_model.pth"
CLASSES = ['fist', 'palm', 'victory'] # dataset.py'de gördüğümüz sıralama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Modeli Kur ve Eğitilmiş Ağırlıkları Yükle
model = GestureCNN(num_classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Modeli kesinlikle test moduna alıyoruz!

# 2. Canlı Görüntü İçin Ön İşleme (Validation Transforms ile aynı olmalı)
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 3. Webcam'i Başlat
cap = cv2.VideoCapture(0)

print("Canlı tahmin başladı! Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1) # Ayna görüntüsü
    
    # ROI (İlgi Alanı) Koordinatları (Veri toplarken kullandığımızın BİREBİR aynısı)
    x1, y1 = 300, 100
    x2, y2 = 550, 350
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Kutu içindeki görüntüyü kes
    roi = frame[y1:y2, x1:x2]
    
    # OpenCV BGR formatını, PyTorch/PIL'in beklediği RGB formatına çevir
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(roi_rgb)
    
    # Görüntüyü tensöre çevir ve boyutunu ayarla: [3, 64, 64] -> [1, 3, 64, 64]
    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    
    # 4. Model Tahmini (Inference)
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Softmax ile sayıları 0-1 arası yüzdelere çevir
        probabilities = F.softmax(outputs, dim=1)
        
        # En yüksek yüzdeyi (confidence) ve onun indeksini (predicted_class) al
        confidence, predicted_class = torch.max(probabilities, 1)
        
    # Yüzdeyi hesapla ve etiket metnini bul
    confidence_pct = confidence.item() * 100
    label = CLASSES[predicted_class.item()]
    
    # Yüzdeyi hesapla
    confidence_pct = confidence.item() * 100
    
    # Arka plandaki isimleri ekranda görünmesini istediğimiz isimlere çevir
    DISPLAY_MAP = {
        'fist': 'ROCK',
        'palm': 'PAPER',
        'victory': 'SCISSORS'
    }
    
    if confidence_pct > 85.0:
        internal_label = CLASSES[predicted_class.item()] # Modelin bildiği isim (örn: 'fist')
        screen_label = DISPLAY_MAP[internal_label]       # Ekrana yazılacak isim (örn: 'ROCK')
        text_color = (0, 255, 0)
    else:
        screen_label = "UNKNOWN"
        text_color = (0, 0, 255)
    
    # Sonuçları Ekrana Yazdır (Artık screen_label kullanıyoruz)
    cv2.putText(frame, f"Gesture: {screen_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.putText(frame, f"Confidence: %{confidence_pct:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    
    cv2.imshow("Gerçek Zamanlı AI", frame)

    # 1 milisaniye bekle, görüntüyü yenile ve 'q' tuşuna basılırsa döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# while döngüsü bittiğinde (q'ya basıldığında) kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()