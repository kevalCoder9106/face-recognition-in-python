import cv2

# picture path
imagePath = "photo3.jpg"
# cascade path
cascadePath= "face-recognition-cascade.xml"

# creating haar cascade
faceCascade = cv2.CascadeClassifier(cascadePath)

# reading image
image = cv2.imread(imagePath)
# converting image to gray image
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the image
faces = faceCascade.detectMultiScale(
    grayImage,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(5, 5),
)

print(f"Found {len(faces)} found!")

#  drawing a rectagle around the faces
for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

# displaying image
cv2.imshow("Image",image)
cv2.waitKey(0)