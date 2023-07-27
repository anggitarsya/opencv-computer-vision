import face_recognition
import pickle

all_face_encodings = {}

img1 = face_recognition.load_image_file("image/anggi-small.jpg")
all_face_encodings["Anggi"] = face_recognition.face_encodings(img1)[0]

img2 = face_recognition.load_image_file("image/kinan-small.jpg")
all_face_encodings["Kinan"] = face_recognition.face_encodings(img2)[0]

# Save the face encodings to a file
with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)
