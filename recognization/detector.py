import numpy as np
import cv2
import matplotlib.pyplot as plt
import sqlite3


faceDetect = cv2.CascadeClassifier('E:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

rec = cv2.face.LBPHFaceRecognizer_create()

rec.read("E:\\Git Folders\\OpenCV\\recognization\\recognizer\\trainingdata.yml")

id = 0
def getProfile(id):

    conn = sqlite3.connect("E:\\Git Folders\\OpenCV\\recognization\\FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID =" + str(id)

    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row[1]
        print(profile)
    conn.close()
    return profile


    
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,img = cam.read()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceDetect.detectMultiScale(gray_image, 1.3, 5)

    
    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

        id, conf = rec.predict(gray_image[y:y+h, x:x+w])

        profile = getProfile(id)
        if(profile != None):
            cv2.putText(img,str(profile[1]), (x, y+h+30), font, 1, (0,0,255),2,cv2.LINE_AA)
            cv2.putText(img,str(profile[2]), (x, y+h+60), font, 1, (0,0,255),2,cv2.LINE_AA)
            cv2.putText(img,str(profile[3]), (x, y+h+90), font, 1, (0,0,255),2,cv2.LINE_AA)
        
        else:

            cv2.putText(img,'Unknown', (x, y+h+30), font, 1, (0,0,255),2,cv2.LINE_AA)

    cv2.imshow("Face",img)

    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
