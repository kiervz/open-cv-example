import cv2 as cv

# Load an image from a file and display it on the screen
group = cv.imread('resources/images/group1.jpg')
cv.imshow('Group of people', group)

# Convert the loaded image from BGR (default OpenCV format) to grayscale
group_gray = cv.cvtColor(group, cv.COLOR_BGR2GRAY)
cv.imshow('Group of people Gray', group_gray)

# Create a Haar cascade classifier for face detection
haar_cascade = cv.CascadeClassifier('./resources/data/haar_face.xml')

# Detect faces in the grayscale image using the Haar cascade classifier
# scaleFactor and minNeighbors are parameters that control the sensitivity 
# and accuracy of the face detection algorithm
# scaleFactor - A value of 1.1 means that the image size is reduced by 10% at each scale. 
# minNeighbors - A value of 1 means that a candidate rectangle needs to have at least one
# neighbor rectangle to be considered a valid face
faces_rect = haar_cascade.detectMultiScale(group_gray, scaleFactor=1.1, minNeighbors=1)

# Print the number of faces detected
print(f'Number of faces found: {len(faces_rect)}')

# Draw a rectangle around each detected face in the original color image
for (x, y, w, h) in faces_rect:
    cv.rectangle(group, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    
# Display the color image with the detected faces on the screen
cv.imshow('Detected Faces', group)

# Wait for a key press to exit the program
cv.waitKey(0)