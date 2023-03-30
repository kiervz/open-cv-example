# import necessary packages
import cv2
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')
cv2.imshow('Blank', blank)

# # 1. Paint the image a certain colour
# blank[height, width]
blank[0:500, 0:250] = (0,0,255)
blank[0:500, 250:500] = (255,0,0)
cv2.imshow('Red and Blue', blank)

# 2. Draw a rectangle
# Link of sample output: http://www.learningaboutelectronics.com/images/Unfilled-rectangle-Python-OpenCV.png 
# cv2.rectangle(blank, (250, 0), (0, 250), (0,255,0), thickness=2)
# cv2.imshow('Rectangle', blank)

# 3. Draw a circle
# cv2.circle(blank, (250, 250), 50, (11, 225, 225), thickness=-1)
# cv2.imshow('Circle', blank)

# 4. Draw a line
# cv2.line(blank, (0, 0), (250, 250), (255,255,255), thickness=3)
# cv2.line(blank, (500, 0), (250, 250), (255,255,255), thickness=3)
# cv2.line(blank, (250, 500), (250, 250), (255,255,255), thickness=3)
# cv2.imshow('Line', blank)

# 5. Write text
# cv2.putText(blank, 'Kamusta, mundo!', (100, 300), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,2), thickness=2)
# cv2.imshow('Text', blank)

# 6. Philippine Flag
# define the vertices of the triangle
pt1 = (0, 0)
pt2 = (500, 0)
pt3 = (250, 250)

# draw the lines that connect the vertices
cv2.line(blank, pt1, pt2, (225, 255, 255), thickness=2)
cv2.line(blank, pt2, pt3, (225, 255, 255), thickness=2)
cv2.line(blank, pt3, pt1, (225, 255, 255), thickness=2)

# define the vertices of the filled triangle as an array
pts = np.array([pt1, pt2, pt3], np.int32)

# fill the triangle with yellow color
cv2.fillPoly(blank, [pts], (225, 255, 255))
cv2.imshow('Philippine Flag', blank)

# 7. Draw 3 stars and a sun
cv2.circle(blank, (250, 90), 40, (11, 225, 225), thickness=-1)
cv2.imshow('Sun', blank)

def createStar(xAxis = 250, yAxis = 250):
    # define the center point of the star
    center = (xAxis, yAxis)

    # define the outer radius and inner radius of the star
    outer_radius = 15
    inner_radius = 5

    # calculate the angles for the five points of the star
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)

    # calculate the coordinates for the outer and inner points of the star
    outer_pts = np.array([(int(center[0] + outer_radius*np.cos(a)), int(center[1] + outer_radius*np.sin(a))) for a in angles], np.int32)
    inner_pts = np.array([(int(center[0] + inner_radius*np.cos(a + np.pi/5)), int(center[1] + inner_radius*np.sin(a + np.pi/5))) for a in angles], np.int32)

    # draw the lines connecting the points of the star
    for i in range(5):
        cv2.line(blank, outer_pts[i], inner_pts[i], (11, 225, 225), thickness=2)
        cv2.line(blank, inner_pts[i], outer_pts[(i+1)%5], (11, 225, 225), thickness=2)

    # display the resulting image
    cv2.imshow('3 Stars', blank)

createStar(250, 210)
createStar(40, 20)
createStar(460, 20)

cv2.waitKey(0)