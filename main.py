import cv2
from sklearn.decomposition import PCA
import math
from jellemzok_kinyerve import adatbazis_jellemzok_lista

def keresett_arc_jellemzok(kep):
    """
    Kivonja a jellemzőket a bemeneti képről.

    Args:
        kep: A feldolgozandó kép.

    Returns:
        list: A kép jellemzői.
    """

    try:
        for (x, y, w, h) in faces_detect:
            face_extract = gray[y:y + h, x:x + w]
            arckep_meret = (100, 100)
            arckep = cv2.resize(face_extract, arckep_meret)
            arckep_vektor = arckep.flatten()
            pca.fit(arckep_vektor.reshape(-1, 1))
            transzformalt_arckep = pca.transform(arckep_vektor.reshape(-1, 1))
            jellemzok = transzformalt_arckep.flatten()  # Jellemzők tárolása
            return jellemzok
        
    except IndexError:
        print("Nincs arc a képen.")

    return None

def euklideszi_tavolsag(vektor1, vektor2):
    """
    Euklidészi távolság kiszámítása két jellemzővektor között.

    Args:
        jellemzok1 (list): Az első jellemzővektor.
        jellemzok2 (list): A második jellemzővektor.

    Returns:
        float: Az euklidészi távolság.
    """
    tavolsag = 0
    for i in range(len(vektor1)):
        tavolsag += (vektor1[i] - vektor2[i])**2
    return math.sqrt(tavolsag)

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
    faces_detect = face_cascade_default.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6)

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

# Minden adatbázisbeli képhez kiszámítja az euklidészi távolságot
for adatbazis_jellemzok in adatbazis_jellemzok_lista:
    tavolsag = euklideszi_tavolsag(jellemzok, adatbazis_jellemzok)
    print(f"Euklidészi távolság: {tavolsag}")


cap.release()
cv2.destroyAllWindows()


