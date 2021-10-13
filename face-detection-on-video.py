import cv2

cascadePath = "face-recognition-cascade.xml"
cascade = cv2.CascadeClassifier(cascadePath)

video = cv2.VideoCapture("src/video.mp4")

while True:
    ret, frame = video.read()

    grayFrame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    
    faces = cascade.detectMultiScale(
        grayFrame,
        1.1,
        10
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()