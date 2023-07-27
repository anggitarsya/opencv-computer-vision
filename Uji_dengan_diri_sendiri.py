import cv2
import face_recognition

def resizedulu(imgfile):
    img = cv2.imread(imgfile)
    img = cv2.resize(img, (900, 1100))
    cv2.imwrite(imgfile, img)

resizedulu("image/anggi-small.jpg")
resizedulu("image/anggi-small.jpg")

known_image = face_recognition.load_image_file("image/kinan-small.jpg")
unknown_image = face_recognition.load_image_file("image/anggi-small.jpg")

known_encoding = face_recognition.face_encodings(known_image)
unknown_encoding = face_recognition.face_encodings(unknown_image)

if len(known_encoding) > 0 and len(unknown_encoding) > 0:
    keanu_encoding = known_encoding[0]
    unknown_encoding = unknown_encoding[0]
    results = face_recognition.compare_faces([keanu_encoding], unknown_encoding)
    print(results)
else:
    print("[False]")
