# Importing necessary libraries
import cv2
import numpy as np

# Reading the image file
image = cv2.imread('resources/images/cats.jpg')
# Displaying the original image
cv2.imshow('Cats', image)

# Creating a blank image with the same dimensions as the original image
blank = np.zeros(image.shape, dtype='uint8')
# Displaying the blank image
cv2.imshow('Blank', blank)

# Converting the original image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Displaying the grayscale image
cv2.imshow('Gray', gray)

# Applying Gaussian blur to the grayscale image
blur = cv2.GaussianBlur(gray, (5,5), cv2.BORDER_DEFAULT)
# Displaying the blurred image
cv2.imshow('Blur', blur)

# Detecting edges using Canny edge detection
canny = cv2.Canny(blur, 125, 175)
# Displaying the detected edges
cv2.imshow('Canny Edges', canny)

# Applying thresholding to the grayscale image
# Note:
# cv.threshold() - This function is used to convert a grayscale image into a binary image.
# 125 - Is the threshold value. And 255 is the maximum value.
# cv2.THRESH_BINARY - Is a type of thresholding method. 
# The cv.THRESH_BINARY method sets all pixels with a value greater than
# the threshold to the maximum value of 255, and all other pixels to 0.
ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
# Displaying the thresholded image
cv2.imshow('Thresh', thresh)

# Finding the contours in the Canny edge-detected image
# cv2.RETR_LIST - specifies the retrieval mode of the contours. 
# It indicates that all the contours should be retrieved and stored in a list.
# cv2.CHAIN_APPROX_SIMPLE - is the contour approximation method. 
# It approximates the contours' shapes by removing any unnecessary points to reduce the storage required. 
# The contours' shapes are stored as a list of boundary points.
contours, hierarchies = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# Printing the number of contours found
print(f'{len(contours)} contour(s) found!')

# Drawing the contours on the blank image
# Note: 
# blank - The image on which the contours will be drawn.
# contours - The contours that will be drawn on the image.
# -1 - Is the index of the contour to be drawn. A value of -1 means that all the contours will be drawn.
# 1 - Is the thickness of the contour line in pixels.
cv2.drawContours(blank, contours, -1, (0, 0, 255), 1)
# Displaying the image with the contours drawn
cv2.imshow('Contours Drawn', blank)

# Waiting for any key press to close all windows
cv2.waitKey(0)


#### Notes: ####

###### Canny image - A Canny image is a binary image that represents the edges in an original image, 
# created using the Canny edge detection algorithm.
# it's a popular edge detection algorithm due to its accuracy and ability to provide thin and 
# continuous edges without false positives. 
# The Canny image can be further processed or analyzed to extract important features from the original image.
###### Contour - Contour is like drawing the shape of something you see in a picture.
# In this case, the computer is tracing the outline of the cats in the picture so that we can see their shapes better.
