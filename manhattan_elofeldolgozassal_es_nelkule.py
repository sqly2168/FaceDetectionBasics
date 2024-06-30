import cv2
import numpy as np
from sklearn.decomposition import PCA
import math
from jellemzok_kinyerve import adatbazis_jellemzok_lista

import cv2
import numpy as np

def kepmegjelenites(filename, jellemzok):
    adatbazis_jellemzok = None
    for elem in adatbazis_jellemzok_lista:
        if elem[0] == filename:
            adatbazis_jellemzok = np.array(elem[1:], dtype=np.float32).reshape((100, 100))
            break

    if adatbazis_jellemzok is not None:
        # Kép javítása kontraszt és fényerő szempontjából
        adatbazis_jellemzok = cv2.normalize(adatbazis_jellemzok, None, 0, 255, cv2.NORM_MINMAX)
        adatbazis_jellemzok = cv2.equalizeHist(adatbazis_jellemzok.astype(np.uint8))

        # Javított kamerakép jellemzők
        kamerakep = np.array(jellemzok, dtype=np.float32).reshape((100, 100))
        kamerakep = cv2.normalize(kamerakep, None, 0, 255, cv2.NORM_MINMAX)
        kamerakep = cv2.equalizeHist(kamerakep.astype(np.uint8))

        cv2.imshow("a legkozelebbi talalat:", adatbazis_jellemzok)
        cv2.imshow("a kamerakep:", kamerakep)
        
        euklideszi_tavolsag_fejlesztett = manhattan_tavolsag(kamerakep.flatten(), adatbazis_jellemzok.flatten())
        print(f"manhattan (elofeldolgozassal): {euklideszi_tavolsag_fejlesztett}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Nincs ilyen nevű kép az adatbázisban: {filename}")


def keresett_arc_jellemzok(kep):
    try:
        faces_detect = face_cascade_default.detectMultiScale(gray, scaleFactor=1.09, minNeighbors=8)
        for (x, y, w, h) in faces_detect:
            face_extract = kep[y:y + h, x:x + w]
            arckep_meret = (100, 100)  # Azonos méret
            arckep = cv2.resize(face_extract, arckep_meret)
            arckep = cv2.cvtColor(arckep, cv2.COLOR_BGR2GRAY)  # Szürkeárnyalatos átalakítás
            arckep = cv2.normalize(arckep, None, 0, 255, cv2.NORM_MINMAX)  # Normalizálás
            arckep = cv2.equalizeHist(arckep)  # Hisztogram kiegyenlítése

            arckep_vektor = arckep.flatten()
            pca.fit(arckep_vektor.reshape(-1, 1))
            transzformalt_arckep = pca.transform(arckep_vektor.reshape(-1, 1))
            return transzformalt_arckep.flatten()  # Visszaadott jellemzők
        
    except IndexError:
        print("Nincs arc a képen.")

    return None


def manhattan_tavolsag(vektor1, vektor2):
    # Biztosítjuk, hogy mindkét vektor azonos hosszúságú és csak numerikus elemeket tartalmaz
    if len(vektor1) != len(vektor2):
        raise ValueError("A vektorok hosszúsága nem egyezik.")
    
    # Konvertálás lebegőpontos típusra, ha szükséges
    vektor1 = np.array(vektor1, dtype=np.float64)
    vektor2 = np.array(vektor2, dtype=np.float64)
    
    # Manhattan-távolság kiszámítása
    különbségek = np.abs(vektor1 - vektor2)  # Abszolút különbség
    tavolsag = np.sum(különbségek)  # Az abszolút különbségek összegzése
    
    return tavolsag


# Kaszkád fájl betöltése
face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# kamera definiálása (legtöbb gépnél a kamera indexe 0)
cap = cv2.VideoCapture(0)

pca = PCA(n_components=1)  # PCA objektum létrehozása

jellemzok = None  # Globális változó - mivel csak egyszer szeretném eltárolni az egyszerűbb összehasonlítás kedvéért


while True:
    # Kép beolvasása
    ret, kep = cap.read()
    gray = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY)
    faces_detect = face_cascade_default.detectMultiScale(gray, scaleFactor=1.09, minNeighbors=8)

    if jellemzok is None:
        jellemzok = keresett_arc_jellemzok(kep) #jellemzok kinyerése csak az első képen
        print(f"Arckép jellemzők: {jellemzok}")

    for (x, y, w, h) in faces_detect: 
        cv2.rectangle(kep, (x, y), (x + w, y + h), (0, 0, 0), 2)

    # Kép megjelenítése
    cv2.imshow("Arcfelismerő valós időben", kep)
    # Kilépés a 'q' billentyű lenyomásával
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#-----
legkozelebbi_talalat_neve = None
legkisebb_tavolsag = float(320000)

for adatbazis_jellemzok in adatbazis_jellemzok_lista:
    filename = adatbazis_jellemzok[0]  # Az első elem a fájlnév

    #--------------------
    #--------------------
    
    tavolsag = manhattan_tavolsag(jellemzok, adatbazis_jellemzok[1:])
    print(f"A kamerakép & {filename} kép ----- Manhattan (sima): {tavolsag}")
    kepmegjelenites(filename, jellemzok)
    if tavolsag < legkisebb_tavolsag:
        legkisebb_tavolsag = tavolsag
        legkozelebbi_talalat_neve = filename

print(f"A legközelebbi kép neve: {legkozelebbi_talalat_neve}")
kepmegjelenites(legkozelebbi_talalat_neve, jellemzok)
#-----



cap.release()
cv2.destroyAllWindows()


