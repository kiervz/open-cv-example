import cv2 as cv

group = cv.imread('resources/images/group1.jpg')
cv.imshow('Group of people', group)

group_gray = cv.cvtColor(group, cv.COLOR_BGR2GRAY)
cv.imshow('Group of people Gray', group_gray)

haar_cascade = cv.CascadeClassifier('./resources/data/haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(group_gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found: {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(group, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    
cv.imshow('Detected Faces', group)

cv.waitKey(0)