import cv2
from sklearn.decomposition import PCA

# Kaszkád fájl betöltése
face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# kamera definiálása (legtöbb gépnél a kamera indexe 0)
cap = cv2.VideoCapture(0)

while True:
    # Kép beolvasása
    ret, kep = cap.read()

    # Szürkeárnyalatos képpé alakítja
    gray = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY)

    # Arcok észlelése
    faces_detect = face_cascade_default.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    try:
        for (x, y, w, h) in faces_detect:
            face_extract = gray[y:y + h, x:x + w]  # Arc kivágása
            arckep_meret = (100, 100)  # Arckép mérete
            arckep = cv2.resize(face_extract, arckep_meret)  # Arckép átméretezése
            arckep_vektor = arckep.flatten()  # Kép vektorizálása
            pca = PCA(n_components=1)  # PCA objektum létrehozása
            pca.fit(arckep_vektor.reshape(-1, 1))  # PCA illesztése az arckép vektorra
            transzformalt_arckep = pca.transform(arckep_vektor.reshape(-1, 1))  # Transzformált arckép
            jellemzok = transzformalt_arckep.flatten()  # Jellemzők kinyerése

            print(f"Arckép jellemzők: {jellemzok}")

            # Arcok rajzolása téglalapban (kék)
            cv2.rectangle(kep, (x, y), (x + w, y + h), (255, 0, 0), 2)

    except IndexError:
        print("Nincs arc a képen.")

    # Kép megjelenítése
    cv2.imshow("Arcfelismerő valós időben", kep)

    # Kilépés a 'q' billentyű lenyomásával
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
