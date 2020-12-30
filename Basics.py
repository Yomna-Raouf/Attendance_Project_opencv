import cv2
import numpy as np
import face_recognition

# Load images and convert them to RGB
ElonImg = face_recognition.load_image_file('ImageBasic/elon-musk.png')
ElonImg = cv2.cvtColor(ElonImg,cv2.COLOR_BGRA2RGB)

ElonTestImg = face_recognition.load_image_file('ImageBasic/elon-test.jpg')
ElonTestImg = cv2.cvtColor(ElonTestImg,cv2.COLOR_BGRA2RGB)

# Detect Faces
ElonFaceLocation = face_recognition.face_locations(ElonImg)[0]
encodeElonFace = face_recognition.face_encodings(ElonImg)[0]

ElonTestFaceLocation = face_recognition.face_locations(ElonTestImg)[0]
encodeElonTestFace = face_recognition.face_encodings(ElonTestImg)[0]

# Draw rectangle around the face
cv2.rectangle(ElonImg, (ElonFaceLocation[3], ElonFaceLocation[0]), (ElonFaceLocation[1], ElonFaceLocation[2]), (0, 255, 255), 2)
cv2.rectangle(ElonTestImg, (ElonTestFaceLocation[3], ElonTestFaceLocation[0]), (ElonTestFaceLocation[1], ElonTestFaceLocation[2]), (0, 255, 255), 2)


cv2.imshow('Elon Musk', ElonImg)
cv2.imshow('Elon Musk Test', ElonTestImg)
cv2.waitKey(0)

