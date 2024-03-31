import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import cv2
import pyzed.sl as sl

def main():
    # Create a Camera object
    # zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    

    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 30  # Set fps at 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    # Capture 50 frames and stop
    i = 0
    image = sl.Mat()
    depth = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    while i < 1:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image, sl.VIEW.LEFT)
            imgL = image.get_data()
            cv2.imshow("Image", imgL)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            zed.retrieve_image(image, sl.VIEW.RIGHT)
            imgR = image.get_data()
            cv2.imshow("Image", imgR)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # Retrieve depth Mat. Depth is aligned on the left image
            depth_data = depth.get_data()
            cv2.imshow("Depth", depth_data)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
            # disparity = stereo.compute(imgL,imgR)
            # plt.imshow(disparity,'gray')
            # plt.show()
            i += 1

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main() 
