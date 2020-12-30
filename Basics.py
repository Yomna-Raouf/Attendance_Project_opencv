import cv2
import numpy as np
import face_recognition

# Load images and convert them to RGB
ElonImg = face_recognition.load_image_file('ImageBasic/elon-musk.png')
ElonImg = cv2.cvtColor(ElonImg,cv2.COLOR_BGRA2RGB)

ElonTestImg = face_recognition.load_image_file('ImageBasic/elon-test.jpg')
ElonTestImg = cv2.cvtColor(ElonTestImg,cv2.COLOR_BGRA2RGB)

cv2.imshow('Elon Musk', ElonImg)
cv2.imshow('Elon Musk Test', ElonTestImg)
cv2.waitKey(0)

