import face_recognition

image = face_recognition.load_image_file("image/anggi.jpg")
face_locations = face_recognition.face_locations(image)

for face_location in face_locations:
    top, right, bottom, left = face_location
    print("Koordinat wajah:")
    print(f"Top: {top}")
    print(f"Right: {right}")
    print(f"Bottom: {bottom}")
    print(f"Left: {left}")
    print()