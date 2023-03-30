# import necessary packages
import cv2

# Read in an image
image = cv2.imread('resources/images/cats.jpg')
cv2.imshow('Read Image', image)

# Converting image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray)

# Blurring the image
# applies a Gaussian blur to the image with a kernel size of 7x7
blur = cv2.GaussianBlur(image, (7, 7), cv2.BORDER_DEFAULT)
cv2.imshow('Blur', blur)

# Edge Cascade
# applies the Canny edge detection algorithm to the image with low and high thresholds of 125 and 175, respectively
canny = cv2.Canny(image, 125, 175)
cv2.imshow('Canny Edges', canny)

# Dilating the image
# dilates the edges in the image with a kernel size of 7x7 and 3 iterations
# what is iterations=3?
# This means that the dilation operation is applied 3 times to the image, 
# which increases the size of the white regions and fills in small gaps between edges
dilated = cv2.dilate(canny, (7, 7), iterations=3)
cv2.imshow('Dilated Image', dilated)

# Eroding the image
# erodes the edges in the image with a kernel size of 7x7 and 3 iterations
eroded = cv2.erode(dilated, (7, 7), iterations=3)
cv2.imshow('Eroded Image', eroded)

# Resizing the image
# resizes the image to a width of 400 pixels and a height of 250 pixels 
resized = cv2.resize(image, (400, 250), interpolation=cv2.INTER_AREA)
cv2.imshow('Resized Image', resized)

# Cropping the image
# crops a rectangular region of the image with top-left coordinates (50, 200) 
# and bottom-right coordinates (200, 400)
cropped = image[50:200, 200:400]
cv2.imshow('Cropped', cropped)

cv2.waitKey(0)



#### Notes: ####

###### Canny image - A Canny image is a binary image that represents the edges in an original image, 
# created using the Canny edge detection algorithm.
# it's a popular edge detection algorithm due to its accuracy and ability to provide thin and continuous edges without false positives. 
# The Canny image can be further processed or analyzed to extract important features from the original image.
###### Dilated image - A dilated image is an image that has undergone a dilation operation, 
# which is a basic morphological image processing technique that expands the bright regions (foreground) in an image.
# The purpose of creating a dilated image is often to make it easier to identify and analyze objects or features in an image.
###### Eroding Image - Eroding an image is a morphological operation that is used to remove small objects or thin out the boundaries of larger objects in an image.
# Erosion is useful for noise reduction, separating overlapping objects, and separating objects that are too close together. 
# The resulting eroded image can be further processed or analyzed to extract important information or features from the original image.

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