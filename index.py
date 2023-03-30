#pylint:disable=no-member

import cv2 as cv
import numpy as np

image = cv.imread('resources/images/cats.jpg')
cv.imshow('Cats', image)

# Translation function that accepts an image and x,y values for translation
def translate(image, x, y):
    # Creating a 2x3 translation matrix to shift the image
    transMat = np.float32([
        [1, 0, x], 
        [0, 1, y]
    ])
    # Storing the dimensions of the original image
    dimensions = (image.shape[1], image.shape[0])
    
    # Applying the translation transformation to the image using the warpAffine function
    # This creates a new image that has been translated by x and y pixels
    translated_image = cv.warpAffine(image, transMat, dimensions)
    
    # Return the translated image
    return translated_image

# -x --> Left
# -y --> Up
# x --> Right
# y --> Down

# Call the translate function with the original image and the x, y translation values
translated = translate(image, 0, -50)
cv.imshow('Translated', translated)

# Rotation
def rotate(image, angle, rotPoint=None):
    (height,width) = image.shape[:2]
    
    # If rotation point is not specified, set it to the center of the image
    if rotPoint is None:
        # The double forward slash (//) operator is used in Python to perform integer division.
        rotPoint = (width//2,height//2)
    
    # Get the rotation matrix for the given angle and rotation point
    # the third argument 1.0 is a scale factor
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    
    # Set the dimensions for the output image
    dimensions = (width,height)

    # Apply the rotation to the image using the warpAffine function
    rotated_image = cv.warpAffine(image, rotMat, dimensions)
    
    # Return the rotated image
    return rotated_image

rotated = rotate(image, 45)
cv.imshow('Rotated 45 degree', rotated)

rotated_rotated = rotate(image, 90)
cv.imshow('Rotated Rotated 90 degree', rotated_rotated)

# Resizing
# resize the image to (250,300) using the INTER_CUBIC interpolation method
resized = cv.resize(image, (250,300), interpolation=cv.INTER_CUBIC)
# display the resized image
cv.imshow('Resized', resized)

# Flipping
# Flipcode:
# 0 - flip vertically (i.e. around the x-axis)
# 1 - flip horizontally (i.e. around the y-axis)
# -1 - flip both vertically and horizontally (i.e. around both axes)
flip = cv.flip(image, -1)
cv.imshow('Flip', flip)

# Cropping
# Selects a specific rectangular region in the image, starting from the top-left pixel at row 200 
# and column 300, and ending at the bottom-right pixel at row 400 and column 400.
cropped = image[200:400, 300:400]
cv.imshow('Cropped', cropped)


cv.waitKey(0)


###### Some of the interpolation you can use:
###### INTER_AREA - Used for reducing the size of an image. 
# It calculates the pixel value of the output image by averaging the pixel values of the input image. 
# This method is a good choice when you need to preserve fine details.
###### INTER_LINEAR - A commonly used method for resizing images. 
# It uses a weighted average of the four closest pixels to determine the value of the output pixel. 
# This method is good for general-purpose image resizing.
###### INTER_CUBIC - Used for increasing the size of an image. 
# It uses a bicubic interpolation method to determine the value of the output pixel, 
# which means that it uses a weighted average of the 16 closest pixels to the output pixel. 
# This method is a good choice when you need to upscale an image while preserving its fine details.