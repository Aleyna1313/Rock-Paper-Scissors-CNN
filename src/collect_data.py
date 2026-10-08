import cv2
import os

# --- AYARLAR ---
# Hangi sınıf için veri topladığını buradan değiştireceksin.
# Sırasıyla "fist", "palm" ve "victory" yaparak bu kodu 3 kez kullanacağız.
CLASS_NAME = "victory" 
NUM_IMAGES = 250     # Sınıf başına toplayacağımız fotoğraf sayısı

# Fotoğrafların kaydedileceği klasör yolu
SAVE_DIR = os.path.join("..", "dataset", CLASS_NAME)

# Webcam'i başlat (0 genelde laptopun kendi kamerasıdır)
cap = cv2.VideoCapture(0)

count = 0
collecting = False

print(f"Toplanacak Sınıf: {CLASS_NAME.upper()}")
print("Başlamak/Duraklatmak için 's' tuşuna, çıkmak için 'q' tuşuna basın.")

while True:
    # Webcam'den bir kare (frame) oku
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break
        
    # Ayna etkisi: Elini sağa götürdüğünde ekranda da sağa gitmesi için görüntüyü ters çeviriyoruz
    frame = cv2.flip(frame, 1) 
    
    # --- ROI (İlgi Alanı) KOORDİNATLARI ---
    # Görüntünün ortalarında 250x250 piksellik bir kare tanımlıyoruz
    x1, y1 = 300, 100
    x2, y2 = 550, 350
    
    # Ekrana bu kutuyu yeşil renk (0, 255, 0) ile çizdiriyoruz
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Ekrana bilgi yazılarını ekleyelim
    cv2.putText(frame, f"Class: {CLASS_NAME.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Count: {count}/{NUM_IMAGES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    status = "Recording..." if collecting else "Paused (Press 's' to start)"
    color = (0, 0, 255) if collecting else (0, 255, 0)
    cv2.putText(frame, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Görüntüyü ekranda göster
    cv2.imshow("Data Collection", frame)
    
    # --- GÖRÜNTÜYÜ KIRPMA VE KAYDETME ---
    # Sadece ROI kutusunun içini dilimliyoruz (Önce Y ekseni, sonra X ekseni yazılır)
    roi = frame[y1:y2, x1:x2]
    
    # Klavyeden basılan tuşu oku
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'): # q'ya basılırsa çık
        break
    elif key == ord('s'): # s'ye basılırsa kaydetmeyi başlat/durdur
        collecting = not collecting 
        
    # Eğer kayıt modundaysak ve hedef sayıya ulaşmadıysak fotoğrafı kaydet
    if collecting and count < NUM_IMAGES:
        img_path = os.path.join(SAVE_DIR, f"{count}.jpg")
        cv2.imwrite(img_path, roi) # ROI matrisini diske .jpg olarak yazdır
        count += 1
        
    if count >= NUM_IMAGES:
        print(f"{CLASS_NAME.upper()} için veri toplama tamamlandı!")
        break

# İşlem bitince kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()