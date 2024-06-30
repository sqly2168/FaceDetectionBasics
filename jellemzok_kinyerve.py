import cv2
from sklearn.decomposition import PCA
import os
import csv
import numpy as np

pca = PCA(n_components=1)
face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Mappa, amelyben a képek találhatók
adatbazis_mappa = "C:\\Users\\SQLY\\Desktop\\egyi\\4F\\FacialRecognition_Basics\\szemelyek"

# Lista a jellemzők tárolására hogy később kiirassuk .csv fájlba
adatbazis_jellemzok_lista = []
detektalt_arcok_szama = 0

for filename in os.listdir(adatbazis_mappa):
    if filename.endswith(".jpg"):
       
        img_path = os.path.join(adatbazis_mappa, filename)
        
        kep = cv2.imread(img_path)
        gray1 = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray1, None, 0, 255, cv2.NORM_MINMAX)  # Normalizálás
        gray = cv2.equalizeHist(gray)  # Hisztogram kiegyenlítése
        faces_detect = face_cascade_default.detectMultiScale(gray, scaleFactor=1.09, minNeighbors=8)
        # Hibakeresés: Ellenőrizze a faces_detect változót
        '''
        #--------------------
        fekete_maszk = cv2.inRange(gray, 0, 50)
        feher_maszk = cv2.inRange(gray, 205, 255)
        szurke_maszk = cv2.inRange(gray, 51, 204)
        teljes_pixelszam = (gray).size
        fekete_pixelek = np.count_nonzero(fekete_maszk)
        fekete_arany = fekete_pixelek / teljes_pixelszam * 100
        feher_pixelek = np.count_nonzero(feher_maszk)
        feher_arany = feher_pixelek / teljes_pixelszam * 100
        szurke_pixelek = np.count_nonzero(szurke_maszk)
        szurke_arany = szurke_pixelek / teljes_pixelszam * 100  

        print(f"Fekete arány: {fekete_arany:.2f}%")
        print(f"Fehér arány: {feher_arany:.2f}%")
        print(f"Szürke arány: {szurke_arany:.2f}%")
        #--------------------'''
        if len(faces_detect) == 0:
            print(f"Nincs arc a(z) {filename} képen.")
        else:
            detektalt_arcok_szama += len(faces_detect)
            for (x, y, w, h) in faces_detect:
                face_extract = gray[y:y + h, x:x + w]
                arckep_meret = (100, 100)
                arckep = cv2.resize(face_extract, arckep_meret)
                arckep_vektor = arckep.flatten()
                pca.fit(arckep_vektor.reshape(-1, 1))  
                transzformalt_arckep = pca.transform(arckep_vektor.reshape(-1, 1))
                adatbazis_jellemzok = transzformalt_arckep.flatten()  

                # Az egyedi azonosító (index) hozzáadása az arckép jellemzők mellé
                adatbazis_jellemzok_lista.append([filename] + adatbazis_jellemzok.tolist()) 

print(f"Összesen {detektalt_arcok_szama} arcot detektáltunk.")

# Jellemzők elmentése CSV fájlba
with open('adatbazis_jellemzok.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(adatbazis_jellemzok_lista)
    