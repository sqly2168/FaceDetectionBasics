import cv2
from sklearn.decomposition import PCA
import os
import csv

pca = PCA(n_components=1)
face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Mappa, amelyben a képek találhatók
adatbazis_mappa = "C:\\Users\\SQLY\\Desktop\\egyi\\4F\\FacialRecognition_Basics\\szemelyek"

# Lista a jellemzők tárolására hogy később kiirassuk .csv fájlba
adatbazis_jellemzok_lista = []

for filename in os.listdir(adatbazis_mappa):
    if filename.endswith(".jpg"):
       
        img_path = os.path.join(adatbazis_mappa, filename)
        
        kep = cv2.imread(img_path)
        gray = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY)
        faces_detect = face_cascade_default.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # Hibakeresés: Ellenőrizze a `faces_detect` változót
        if len(faces_detect) == 0:
            print(f"Nincs arc a(z) {filename} képen.")
        else:
            for (x, y, w, h) in faces_detect:
                face_extract = gray[y:y + h, x:x + w]
                arckep_meret = (100, 100)
                arckep = cv2.resize(face_extract, arckep_meret)
                arckep_vektor = arckep.flatten()
                pca.fit(arckep_vektor.reshape(-1, 1))  
                transzformalt_arckep = pca.transform(arckep_vektor.reshape(-1, 1))
                adatbazis_jellemzok = transzformalt_arckep.flatten()  

                adatbazis_jellemzok_lista.append(adatbazis_jellemzok) 

# Jellemzők elmentése CSV fájlba
with open('adatbazis_jellemzok.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(adatbazis_jellemzok_lista)

            
