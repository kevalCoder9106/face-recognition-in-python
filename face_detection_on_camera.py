import cv2
from face_detection import face_detection

# creating face_detection module instance
face_detection_inst = face_detection()

# referencing camera
camera = cv2.VideoCapture(1)

# displaying live camera footage with faces detected in it
while True:
    # getting a frame from video
    ret, frame = camera.read()

    # return faces in frame
    faces = face_detection_inst.detect_faces(frame)

    # drawing rectangle in original frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # displaying the frame
    cv2.imshow("Frame", frame)

    # waiting for q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
