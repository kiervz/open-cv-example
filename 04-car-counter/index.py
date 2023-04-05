# Usage: python index.py --algo=nano --issave=true

import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--algo", help="Specify the YOLO Algorithm",
                choices=['nano', 'small', 'medium', 'large', 'extra'])

ap.add_argument("-is", "--issave", required=True, help="Save each video frame as JPEG image?")
args = vars(ap.parse_args())

# Load the video file
cap = cv2.VideoCapture('./videos/cars.mp4')

yolo_algorithm = {
    'nano': 'n',
    'small': 's',
    'medium': 'm',
    'large': 'l',
    'extra': 'x'
}

if not args['algo']:
    args['algo'] = 'nano'

# Load the YOLO object detection model
model = YOLO(f"./weights/yolov8{yolo_algorithm[args['algo']]}.pt")

# List of class names for the objects that the model can detect
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
              "tie", "suitcase", "frisbee", "skis", "snowboard sports ball", "kate", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "Laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven ", "toaster", "sink", "refrigerator", "book", "clock", "vase ", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load the mask image
mask = cv2.imread('./images/mask.png')

# Create a SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Set the limits of the counting line
# x1, y1, x2, y2
limits = [373, 290, 673, 290]

# List to keep track of the IDs of all vehicles that have crossed the counting line
totalCounts = []

# Frame counter
frame_num = 0

# Loop through each frame of the video
while True:
    # Read a frame from the video
    success, img = cap.read()

    # Stop if no more frames are available
    if not success:
        break

    # Apply the mask to the image to isolate the region of interest
    imgRegion = cv2.bitwise_and(img, mask)

    # Detect objects in the region of interest using the YOLO model
    results = model(imgRegion, stream=True)

    # Load the graphic image
    # the cv2.IMREAD_UNCHANGED flag ensures that the original image data, including
    # any transparency information, is preserved when reading the image with OpenCV.
    imageGraphic = cv2.imread('./images/graphics.png', cv2.IMREAD_UNCHANGED)

    # Overlay the graphic image onto the frame
    img = cvzone.overlayPNG(img, imageGraphic, (0, 0))

    # Create an empty array to store the bounding boxes, confidence scores, and class IDs of all vehicles
    detections = np.empty((0, 5))

    # Loop through each detected object
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0]

            # Convert to integer values
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Get the width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Extract the confidence score for the detection and round it to two decimal places
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Extract the class ID for the detected object
            cls = int(box.cls[0])

            # Extract the class name for the detected object
            currentClass = classNames[cls]

            # If currentClass is equal to 'car', 'truck', or 'bus'
            # and the confidence is greater than 30%
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" and conf > 0.3:
                # cvzone.putTextRect(imgRegion, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)

                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])

                # Concatenate the currentArray
                detections = np.vstack((detections, currentArray))

    # Update tracker with the detections and get the updated results
    resultsTracker = tracker.update(detections)

    # Draw a line on the image to represent the boundary for counting objects
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), thickness=5)

    # Loop through the tracked objects
    for result in resultsTracker:
        # Extract the bounding box coordinates and ID for the tracked object
        x1, y1, x2, y2, Id = result
        # Convert to integer values
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)

        # Calculate the width and height of the bounding box
        w, h = x2 - x1, y2 - y1

        # Draw a corner rectangle around the bounding box
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

        # calculate the center point of the bounding box
        cx, cy = x1 + w // 2, y1 + h // 2

        # This line draws a filled circle on the image at the coordinates (cx, cy)
        # with a radius of 5 pixels and a color of magenta (255, 0, 255).
        # cv2.FILLED specifies the thickness of the circle shape to be filled completely
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # This line checks if the x-coordinate of the circle is within the left and right limits
        # and if the y-coordinate of the circle is within 15 pixels above or below the upper limit.
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # Check if the current 'Id' (identity) of the circle has not already been counted before.
            if totalCounts.count(Id) == 0:
                # If the current Id has not been counted before, this line appends the Id to the totalCounts list.
                totalCounts.append(Id)
                # This line draws a green line on the image from the point (limits[0], limits[1])
                # to the point (limits[2], limits[3]) with a thickness of 5 pixels.
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), thickness=5)

        # Add a text label to the image indicating the total count of unique identities counted so far.
        # The text is centered at the coordinates (240, 80) and has a black background color (colorT)
        # with a light pink text color (colorR).
        cvzone.putTextRect(img, f'{len(totalCounts)}', (240, 80), colorT=(0, 0, 0), colorR=(255, 233, 193))

    # If issave argument is equal to true, save each frame
    if args['issave'] == 'true':
        # Save current frame as a JPEG image
        frame_path = f'./images/data/frame_{frame_num:06d}.jpg'
        cv2.imwrite(frame_path, img)

        # Increment frame counter
        frame_num += 1

    cv2.imshow('Image', img)
    # cv2.imshow('ImageRegion', imgRegion)
    cv2.waitKey(1)
