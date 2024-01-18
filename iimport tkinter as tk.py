import cv2
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Modeli yükleme
model_path = "cataract-classification-with-two-retina-datasets"  # eğittiğiniz modelin dosya yolunu buraya yazın
model = load_model(model_path)

# Tkinter penceresi oluşturma
root = tk.Tk()
root.title("Canlı Cataract Tanıma")

# Kamera başlatma
cap = cv2.VideoCapture(0)  # 0, bilgisayarınızda mevcut olan bir kamera demektir

# Canvas oluşturma
canvas = Canvas(root, width=640, height=480)
canvas.pack()

def process_frame():
    # Kameradan bir kare al
    ret, frame = cap.read()

    # Göz tespiti için bir göz tanıma sınıfı kullanabilirsiniz (örneğin, OpenCV'deki CascadeClassifier)
    # Bu sadece bir örnek, daha gelişmiş yöntemler de kullanabilirsiniz
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Gri tonlamaya çevirme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gözleri tespit etme
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in eyes:
        # Göz bölgesini çıkartma
        eye_roi = frame[y:y + h, x:x + w]

        # Göz bölgesini modelinize uygun formata getirme, örneğin boyutlandırma ve normalizasyon
        target_size = (128, 128)
        resized_eye = cv2.resize(eye_roi, target_size)
        normalized_eye = resized_eye / 255.0

        # Modelinize tahmin yaptırma
        processed_eye = np.expand_dims(normalized_eye, axis=0)
        prediction = model.predict(processed_eye)

        # Tahmin sonuçlarını işleme
        if prediction[0][0] > 0.5:
            label = "Cataract"
        else:
            label = "Normal"

        # Göz bölgesini çerçeve içine alma
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Görüntü üzerine etiketi yazdırma
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # OpenCV'den Tkinter'a çerçeve dönüştürme
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_image)
    img_tk = ImageTk.PhotoImage(image=img)

    # Canvas'a resmi yerleştirme
    canvas.img_tk = img_tk
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    # 30 milisaniye sonra fonksiyonu tekrar çağır
    root.after(30, process_frame)

# İlk kareyi almak için fonksiyonu çağır
process_frame()

# Pencereyi kapatma işlevi
def close_window():
    cap.release()
    root.destroy()

# Pencereyi kapatma düğmesi
close_button = tk.Button(root, text="Kapat", command=close_window)
close_button.pack(pady=10)

# Pencereyi açma
root.mainloop()
