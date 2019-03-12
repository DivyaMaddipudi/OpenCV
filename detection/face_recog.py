import numpy as np
import cv2
import matplotlib.pyplot as plt


faceDetect = cv2.CascadeClassifier('E:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)


while(True):
    ret,img = cam.read()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceDetect.detectMultiScale(gray_image, 1.3, 5)

    
    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

    cv2.imshow("Face",img)

    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
