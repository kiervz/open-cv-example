import numpy as np
import cv2 as cv
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")

args = vars(ap.parse_args())
image = cv.imread(args["image"])
cv.imshow('est', image)

haar_cascade = cv.CascadeClassifier('./resources/data/haar_face.xml')
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('label.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect the face in image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with confidence of {confidence}')
    
    cv.putText(image, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=2)
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    
cv.imshow('Detected face', image)
cv.waitKey(0)