# Usage: python index.py --image shapes.png --name new-image --save images-folder
# Note: --save argument is not required

# import necessary packages
import argparse
import cv2
import imghdr
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-n", "--name", required=True,
    help="new name of image")
ap.add_argument("-s", "--save",
    help="save within the folder or to new folder")
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

# get the extension_name of image file 
extension_name = imghdr.what(args['image'])

output_path = "{}.{}".format(args['name'], extension_name)
if args.get('save'):
    # make new directory folder based on -s or --save argument
    os.makedirs(args['save'], exist_ok=True)
    output_path = "{}/{}".format(args['save'], output_path)

# save new image file
cv2.imwrite(output_path, image)