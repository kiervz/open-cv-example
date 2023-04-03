# The purpose of this code is to train a face recognition model using the 
# LBPH algorithm on a set of images
# Usage: python index.py 

# Imports necessary libraries
import os 
import cv2 as cv
import numpy as np

# List of people names
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# Path to the directory where training images are located   
DIR = './resources/images/faces/train'
# Create a Haar cascade classifier for face detection
haar_cascade = cv.CascadeClassifier('./resources/data/haar_face.xml')
p = []

for i in os.listdir('./resources/images/faces/train'):
    p.append(i)

# Two empty lists to store image features and their corresponding labels
features = []
labels = []

# Function to read training images and their labels
def create_train():
    # Loop through each person's name in 'people' list
    for person in people:
        # Construct the path to the directory where images of every person are located
        path = os.path.join(DIR, person)
        # Assign a unique integer label to each person in 'people' list
        label = people.index(person)
        
        # Loop through each image file in the current person's directory
        for img in os.listdir(path):
            # Construct the path to the image file
            img_path = os.path.join(path, img)
            
            # Read the image file in grayscale
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            # Detect faces in the image
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            # Draw a rectangle around each detected face in the original color image
            for (x, y, w, h) in faces_rect:
                # Extract the region of interest (ROI) that contains the face
                faces_roi = gray[y:y+h, x:x+w]
                # Append the ROI to the features list
                features.append(faces_roi)
                # Append the label of the person to the labels list
                labels.append(label)
                
create_train()

print('Training done -------------')

# Convert features and labels lists to numpy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# Create an instance of LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, labels)

# Save the trained recognizer to disk
face_recognizer.save('face_trained.yml')

# Save the features and labels lists to disk
np.save('features.npy', features)
np.save('labels.npy', labels)

# print(f'Length of the features = {len(features)}')
# print(f'Length of the labels = {len(labels)}')

# Note: 
# Region of Interest - ROI helps to focus on a specific part (e.g. Face) of an image for further analysis or processing