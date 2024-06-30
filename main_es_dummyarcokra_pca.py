import cv2
import numpy as np
from sklearn.decomposition import PCA
import math
from jellemzok_kinyerve import adatbazis_jellemzok_lista

def keresett_arc_jellemzok(kep, pca):
    """
    Kivonja a jellemzőket a bemeneti képről.

    Args:
        kep: A feldolgozandó kép.
        pca: A PCA objektum.

    Returns:
        list: A kép jellemzői.
    """
    gray = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY)
    faces_detect = face_cascade_default.detectMultiScale(gray, scaleFactor=1.09, minNeighbors=8)

    for (x, y, w, h) in faces_detect:
        face_extract = gray[y:y + h, x:x + w]
        arckep_meret = (100, 100)
        arckep = cv2.resize(face_extract, arckep_meret)
        arckep_vektor = arckep.flatten()
        transzformalt_arckep = pca.transform(arckep_vektor.reshape(1, -1))
        jellemzok = transzformalt_arckep.flatten()  # Jellemzők tárolása
        return jellemzok
    
    return None

def euklideszi_tavolsag(vektor1, vektor2):
    """
    Euklidészi távolság kiszámítása két jellemzővektor között.

    Args:
        vektor1 (list): Az első jellemzővektor.
        vektor2 (list): A második jellemzővektor.

    Returns:
        float: Az euklidészi távolság.
    """
    return np.linalg.norm(np.array(vektor1) - np.array(vektor2))

# Kaszkád fájl betöltése
face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# kamera definiálása (legtöbb gépnél a kamera indexe 0)
cap = cv2.VideoCapture(0)

pca = PCA(n_components=150)  # PCA objektum létrehozása, adjust n_components as needed

# Dummy adatbázis létrehozása a PCA fittálásához, helyettesítsd az aktuális adatbázisoddal
dummy_faces = np.random.rand(100, 10000)  # 100 dummy arckép vektor mérete 100x100
pca.fit(dummy_faces)  # PCA fittálása a dummy adatokon

# Transzformáljuk az adatbázis jellemzőit a PCA-val
transzformalt_adatbazis_jellemzok_lista = []
for adatbazis_jellemzok in adatbazis_jellemzok_lista:
    arckep_vektor = np.array(adatbazis_jellemzok[1:]).flatten()
    transzformalt_arckep = pca.transform(arckep_vektor.reshape(1, -1))
    transzformalt_adatbazis_jellemzok_lista.append((adatbazis_jellemzok[0], transzformalt_arckep.flatten()))

jellemzok = None  # Globális változó - mivel csak egyszer szeretném eltárolni az egyszerűbb összehasonlítás kedvéért

while True:
    # Kép beolvasása
    ret, kep = cap.read()
    if not ret:
        continue
    
    if jellemzok is None:
        jellemzok = keresett_arc_jellemzok(kep, pca)  # jellemzok kinyerése csak az első képen
        if jellemzok is not None:
            print(f"Arckép jellemzők: {jellemzok}")

    gray = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY)
    faces_detect = face_cascade_default.detectMultiScale(gray, scaleFactor=1.09, minNeighbors=8)

    for (x, y, w, h) in faces_detect: 
        cv2.rectangle(kep, (x, y), (x + w, y + h), (0, 0, 0), 2)

    # Kép megjelenítése
    cv2.imshow("Arcfelismerő valós időben", kep)

    # Kilépés a 'q' billentyű lenyomásával
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Minden adatbázisbeli képhez kiszámítja az euklidészi távolságot
if jellemzok is not None:
    for adatbazis_jellemzok in transzformalt_adatbazis_jellemzok_lista:
        tavolsag = euklideszi_tavolsag(jellemzok, adatbazis_jellemzok[1:])
        print(f"Arckép: {adatbazis_jellemzok[0]}")
        print(f"Euklidészi távolság: {tavolsag}")

cap.release()
cv2.destroyAllWindows()
