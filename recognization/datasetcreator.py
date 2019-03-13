import numpy as np
import cv2
import matplotlib.pyplot as plt
import sqlite3


    
faceDetect = cv2.CascadeClassifier('E:\OpenCV\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)


def insertOrUpdate(Id, Name):
    conn = sqlite3.connect("E:\\Git Folders\\OpenCV\\recognization\\FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID = "+ str(Id)
    cursor = conn.execute(cmd)
    isRecordExit = 0
    for row in cursor:
        isRecordExit = 1

    if(isRecordExit == 1):
        cmd = "UPDATE People SET Name" + str(Name) + "WHERE ID =" + str(Id)

    else:
        cmd = "Insert into People(ID,Name) Values (" +str(Id) + "," +str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()

id = int(input("enter user id"))
name = input("Enter name")

insertOrUpdate(id, name)
sampleNum = 0
while(True):
    ret,img = cam.read()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceDetect.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, w, h) in faces:

        sampleNum = sampleNum + 1

        cv2.imwrite("E:\\Git Folders\\OpenCV\\recognization\\dataSet\\User." + str(id) + "." + str(sampleNum) + ".jpg", gray_image[y:y+h, x:x+w])

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


        cv2.waitKey(100)

    cv2.imshow("Face", img)

    cv2.waitKey(1)

    if sampleNum > 20:
        break
cam.release()
cv2.destroyAllWindows()
