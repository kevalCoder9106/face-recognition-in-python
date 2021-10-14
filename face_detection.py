import cv2


class face_detection:
    def __init__(self, accuracy=1.2, features=6):
        # cascade path
        self.cascadePath = "cascade/face-recognition-cascade.xml"
        # creating haar cascade
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.accuracy = accuracy
        self.feature = features

    def detect_faces(self, orig_image):
        # converting image to gray image
        gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)

        # detect faces in the image
        faces = self.faceCascade.detectMultiScale(
            gray_image,
            self.accuracy,
            self.feature
        )

        return faces


def main():
    print("This module can be used to return face in any image")


if __name__ == "__main__":
    main()
