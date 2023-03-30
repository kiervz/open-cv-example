# import necessary packages
import cv2

# open video capture
capture = cv2.VideoCapture('resources/videos/dog.mp4')

# function to resize frame
def rescaleFrame(frame, scale=0.75):
    # get the original frame dimensions
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    
    # set the new dimensions for the resized frame
    dimensions = (width, height)
    
    # resize the frame using the specified interpolation method
    # (in this case, cv2.INTER_AREA)
    resized_frame = cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
    
    # return the resized frame
    return resized_frame

# loop through video frames
while True:
    # read video frame
    isTrue, frame = capture.read()
    
    # resize video frame
    frame_resized = rescaleFrame(frame)
    
    # show video frames
    cv2.imshow('Video', frame)
    cv2.imshow('Video Rescale', frame_resized)
    
    # press 'd' key to exit loop
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break
    
# release video capture and destroy windows
capture.release()
cv2.destroyAllWindows()