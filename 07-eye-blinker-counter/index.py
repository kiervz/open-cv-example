import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

# ID list of both left and right eyes landmarks
# idList = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374, 373,
#           390, 249, 263, 466, 388, 387, 386, 385,384, 398]

# ID list of left eye landmarks only
id_list = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]

eye_ratio_list = []
BLINK_COUNTER = 0
FRAME_COUNTER = 0
VERIFIED_TOTAL_BLINK = 9

cap = cv2.VideoCapture('./Blink.mp4')
detector = FaceMeshDetector(maxFaces=1)

while True:
    # reset the video capture frame if it reaches the end
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        # get the first detected face since we set maxFaces=1 in FaceMeshDetector
        face = faces[0]

        left_eye_upper = face[159]
        left_eye_lower = face[23]
        left_eye_left = face[130]
        left_eye_right = face[243]

        # find the distance between two landmarks
        horizontal_length, _ = detector.findDistance(left_eye_left, left_eye_right)
        vertical_length, _ = detector.findDistance(left_eye_upper, left_eye_lower)

        # draw line verifically and horizontally
        # cv2.line(img, left_eye_upper, left_eye_lower, (0, 200, 0), 2)
        # cv2.line(img, left_eye_left, left_eye_right, (200, 200, 0), 1)

        eye_ratio = int((vertical_length/horizontal_length)*100)
        eye_ratio_list.append(eye_ratio)

        # if length of eye_ratio_list is greater than 3
        # remove the first index of eye_ratio_list
        if len(eye_ratio_list) > 3:
            eye_ratio_list.pop(0)

        ratio_average = sum(eye_ratio_list)/len(eye_ratio_list)
        # print('ratio_average: ', ratio_average)

        if BLINK_COUNTER != VERIFIED_TOTAL_BLINK:
            if ratio_average < 35 and FRAME_COUNTER == 0:
                BLINK_COUNTER += 1
                FRAME_COUNTER = 1

            if FRAME_COUNTER != 0:
                FRAME_COUNTER += 1
                if FRAME_COUNTER > 10:
                    FRAME_COUNTER = 0

        # print("FRAME_COUNTER: ", FRAME_COUNTER)

    # display blink count
    cvzone.putTextRect(img, f'Blink: {BLINK_COUNTER}', (50, 100), colorR=(255, 0, 255))

    # display verification status if the blink count reaches the VERIFIED_TOTAL_BLINK
    if BLINK_COUNTER == VERIFIED_TOTAL_BLINK:
        cvzone.putTextRect(img, 'Verified', (50, 200), colorR=(0, 255, 0))

    img = cv2.resize(img, (640, 360))
    cv2.imshow('KYC Verification', img)
    key = cv2.waitKey(30)

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()