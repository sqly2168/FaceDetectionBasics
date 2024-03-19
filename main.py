import cv2

# Kaszkád fájl betöltése
face_cascade = cv2.CascadeClassifier('C:\\Users\\SQLY\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

# kamera definiálása (lrgtöbb gépnél a kamera indexe 1)
cap = cv2.VideoCapture(0)

# #képbeolvasás
# kep = cv2.imread("C:\\Users\\SQLY\\Desktop\\ow.png")

while True:
    # Kép beolvasása
    ret, kep = cap.read()

    # Szürkeárnyalatos képpé alakítja
    gray = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY)

    # Arcok észlelése
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Arcok rajzolása téglalapban
    for (x, y, w, h) in faces:
        cv2.rectangle(kep, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Kép megjelenítése
    cv2.imshow("Arcfelismerő valós időben", kep)

    # Kilépés a 'q' billentyű lenyomásával
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
