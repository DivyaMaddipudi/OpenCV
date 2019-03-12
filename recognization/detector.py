import numpy as np
import cv2
import matplotlib.pyplot as plt


faceDetect = cv2.CascadeClassifier('E:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

rec = cv2.face.LBPHFaceRecognizer_create()

rec.read("E:\\Git Folders\\OpenCV\\recognization\\recognizer\\trainingdata.yml")

id = 0

font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,img = cam.read()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceDetect.detectMultiScale(gray_image, 1.3, 5)

    
    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

        id, conf = rec.predict(gray_image[y:y+h, x:x+w])

        if id == 1:
            id = "Divya"
        elif id == 2:
            id = "Vidya"
            
        cv2.putText(img,str(id), (x, y+h), font, 2, (0,0,255),2,cv2.LINE_AA)

    cv2.imshow("Face",img)

    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
