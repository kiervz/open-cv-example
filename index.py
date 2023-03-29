# import necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-n", "--name", required=True,
    help="new name of image")
args = vars(ap.parse_args())

# load the image from disk via "cv2.imread" and then grab the spatial
# dimensions, including width, height, and number of channels.
# Note: In Open CV images are represented as numpy arrays
image = cv2.imread(args["image"])
(h, w, c) = image.shape[:3]

# display the image width, height and number of channels to our 
# terminal
print("width: {} pixels".format(w))
print("height: {} pixels".format(h))
print("channels: {} pixels".format(c))

# show the image and wait for a keypress
cv2.imshow("PySearch", image)
# waitKey is just a pause, it's a waiting for user input
cv2.waitKey(0)