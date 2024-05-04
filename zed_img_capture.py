"""
Script Name: zed_img_capture.py
Description: Test file for playing around with ZED camera settings
Author: Yatharth Ahuja, David Hill, Michael Gromic, Leo Mouta, Louis Plottel
"""

# reference: https://www.stereolabs.com/docs/opencv/python

 # INTRINSICS:
# [LEFT_CAM_2K]
# fx=1399.5
# fy=1399.5
# cx=1168.65
# cy=640.13
# k1=-0.173127
# k2=0.0247692
# p1=0
# p2=0
# k3=0

# [RIGHT_CAM_2K]
# fx=1401.12
# fy=1401.12
# cx=1176.56
# cy=683.206
# k1=-0.173623
# k2=0.0259224
# p1=0
# p2=0
# k3=0

import cv2
import numpy as np

def undistort_fisheye(img):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        center = circles[0, 0, :2]
        radius = circles[0, 0, 2]

        # Undistort the image
        undistorted = cv2.fisheye.undistortImage(img, K=None, D=None, Knew=None)

        # Crop the undistorted image
        x, y = center[0] - radius, center[1] - radius
        width, height = radius * 2, radius * 2
        undistorted_cropped = undistorted[y:y+height, x:x+width]

        return undistorted_cropped
    else:
        print("No circles detected.")
        return None

# Open the ZED camera
cap = cv2.VideoCapture(0)
if cap.isOpened() == 0:
    exit(-1)

# Set the video resolution to HD720 (2560*720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

i = 0
while True:
    # Get a new frame from camera
    retval, frame = cap.read()
    # Extract left and right images from side-by-side
    left_right_image = np.split(frame, 2, axis=1)
    # Display images
    # cv2.imshow("frame", frame)
    # cv2.imshow("right", left_right_image[0])
    cv2.imshow("left", left_right_image[1])
    # cv2.imwrite("left.jpg", left_right_image[1])
    
    # panorama = True
    # if panorama:
    #     # Stitch the left and right images together
    #     stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    #     status, pano = stitcher.stitch([left_right_image[1], left_right_image[0]])
    #     cv2.imshow("panorama", pano)       
    
    # cv2.imshow(undistort_fisheye(left_right_image[1]))
    if cv2.waitKey(30) == ord('s'):
        cv2.imwrite("left_test"+str(i)+".jpg", left_right_image[1])
        i+=1
        continue
    elif cv2.waitKey(30) == ord('q'):
        break

exit(0)
