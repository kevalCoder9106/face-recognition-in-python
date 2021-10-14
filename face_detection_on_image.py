import cv2
from face_detection import face_detection

# creating face_detection module instance
face_detection_inst = face_detection()

# reading image
image = cv2.imread("src/photo3.jpg")

faces = face_detection_inst.detect_faces(image)

# drawing rectangle in original frame
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# displaying the frame
cv2.imshow("Frame", image)

cv2.waitKey(0)
