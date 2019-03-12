import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = "E:\\Git Folders\\OpenCV\\recognization\\dataSet"

def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    IDs = []

    for imagepath in imagePaths:
        faceImg = Image.open(imagepath).convert("L")
        faceNp = np.array(faceImg, 'uint8')

        ID = int(os.path.split(imagepath)[-1].split('.')[1])

        #print(ID)

        faces.append(faceNp)

        IDs.append(ID)

        cv2.imshow("training", faceNp)
        cv2.waitKey(10)

    return np.array(IDs), faces

Ids, faces = getImagesWithID(path)

recognizer.train(faces, Ids)

recognizer.save("E:\\Git Folders\\OpenCV\\recognization\\recognizer\\trainingdata.yml")

cv2.destroyAllWindows()

