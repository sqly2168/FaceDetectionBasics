import cv2

# Kaszkád fájl betöltése (arcfelismerés haar algoritmussal)
#laptop
#face_cascade = cv2.CascadeClassifier('C:\\Users\\SQLY\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

#otthoni pc
face_cascade = cv2.CascadeClassifier('C:\\Users\\sqly2\\PycharmProjects\\FaceDetectionBasics\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

#képbeolvasás
#laptop
#kep = cv2.imread("C:\\Users\\SQLY\\Pictures\\Camera Roll\\Face.jpg")

#otthoni pc
kep = cv2.imread("C:\\Users\\sqly2\\Desktop\\kep.png")

# Szürkeárnyalatos képpé alakítja (szükséges a kaszkád osztályozó számára)
gray = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY)

# Arcok észlelése
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Arcok rajzolása téglalap alakban
for (x, y, w, h) in faces:
    cv2.rectangle(kep, (x, y), (x+w, y+h), (255, 0, 0), 2)

#megjeleniti a képet
cv2.imshow("Arcfelismerő első képe", kep)

#billentyű lenyomással kilép
cv2.waitKey(0)

#bezárja az ablakot
cv2.destroyAllWindows()
