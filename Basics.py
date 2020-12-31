import cv2
import numpy as np
import face_recognition

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# Load images and convert them to RGB and resize it to fit the screen
ElonImg = face_recognition.load_image_file('ImageBasic/elon-musk.png')
ElonImg = ResizeWithAspectRatio(ElonImg, height=1000)
# Resize by width OR
# resize = ResizeWithAspectRatio(image, height=1280) # Resize by height
ElonImg = cv2.cvtColor(ElonImg,cv2.COLOR_BGRA2RGB)

ElonTestImg = face_recognition.load_image_file('ImageBasic/elon-test.jpg')
ElonTestImg = ResizeWithAspectRatio(ElonTestImg, height=1000)
ElonTestImg = cv2.cvtColor(ElonTestImg,cv2.COLOR_BGR2RGB)


BillTestImg = face_recognition.load_image_file('ImageBasic/bill-gates.jpg')
BillTestImg = ResizeWithAspectRatio(BillTestImg, height=1000)
BillTestImg = cv2.cvtColor(BillTestImg, cv2.COLOR_BGR2RGB)


# Detect Faces
ElonFaceLocation = face_recognition.face_locations(ElonImg)[0]
encodeElonFace = face_recognition.face_encodings(ElonImg)[0]

ElonTestFaceLocation = face_recognition.face_locations(ElonTestImg)[0]
encodeElonTestFace = face_recognition.face_encodings(ElonTestImg)[0]

BillTestFaceLocation = face_recognition.face_locations(BillTestImg)[0]
encodeBillTestFace = face_recognition.face_encodings(BillTestImg)[0]


# Draw rectangle around the face
cv2.rectangle(ElonImg, (ElonFaceLocation[3], ElonFaceLocation[0]), (ElonFaceLocation[1], ElonFaceLocation[2]), (0, 255, 255), 2)
cv2.rectangle(ElonTestImg, (ElonTestFaceLocation[3], ElonTestFaceLocation[0]), (ElonTestFaceLocation[1], ElonTestFaceLocation[2]), (0, 255, 255), 2)
cv2.rectangle(BillTestImg, (BillTestFaceLocation[3], BillTestFaceLocation[0]), (BillTestFaceLocation[1], BillTestFaceLocation[2]), (0, 255, 255), 2)

# 128 measurement of the faces
results = face_recognition.compare_faces([encodeElonFace], encodeElonTestFace)
print(results)
results2 = face_recognition.compare_faces([encodeElonFace],encodeBillTestFace)
print(results2)

# Similarity between faces "Best Match"
faceDistance = face_recognition.face_distance([encodeElonFace], encodeElonTestFace)
BillFaceDistance = face_recognition.face_distance([encodeElonFace], encodeBillTestFace)

cv2.putText(ElonTestImg, f'{results} {round(faceDistance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (250, 200, 100), 2, cv2.LINE_8)



# Show Images
cv2.imshow('Elon Musk', ElonImg)
cv2.imshow('Elon Musk Test', ElonTestImg)
cv2.imshow('Bill Gates Test', BillTestImg)
cv2.waitKey(0)

