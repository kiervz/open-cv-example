# Usage: python face_recognition.py --image="./resources/images/faces/val/elton_john/1.jpg"

# Imports necessary libraries
import numpy as np
import cv2 as cv
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# Add an argument to the parser that specifies the path to the input image
ap.add_argument("-i", "--image", required=True,
    help="path to input image")

args = vars(ap.parse_args())
image = cv.imread(args["image"])
cv.imshow('Test', image)

# Load a pre-trained Haar cascade classifier for detecting faces in images
haar_cascade = cv.CascadeClassifier('./resources/data/haar_face.xml')
# Define a list of people's names that correspond to different labels
# that will be output by the face recognition algorithm.
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('label.npy')

# Create a Local Binary Patterns Histograms (LBPH) face recognizer object
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# Load a pre-trained face recognizer from a YAML file
face_recognizer.read('face_trained.yml')

# Convert the input image to grayscale using the 'cvtColor' function
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect faces in the grayscale image using the Haar cascade classifier.
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    # Extract the region of interest (ROI) in the grayscale image that corresponds to the detected face
    faces_roi = gray[y:y+h, x:x+w]
    
    # Use the face recognizer to predict the label (person's name) and confidence score for the ROI
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with confidence of {confidence}')
    
    # Output the predicted label on the input image using the 'putText' function
    cv.putText(image, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=2)
    
    # Draw a rectangle around the detected face on the input image using the 'rectangle' function
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    
cv.imshow('Detected face', image)
cv.waitKey(0)