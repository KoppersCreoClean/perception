"""
Script Name: zed_img_capture.py
Description: Test file for playing around with ZED camera settings using ZED SDK
Author: Yatharth Ahuja, David Hill, Michael Gromic, Leo Mouta, Louis Plottel
"""

import pyzed.sl as sl

def zed_cap_image():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 30  # Set fps at 30

   

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    # grab an image, a RuntimeParameters object must be given to grab()
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # zed.retrieve_image(image, sl.VIEW.RIGHT)
        
        image_cv = image.get_data()
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)
        
    # Close the camera
    zed.close()

    return image_cv, timestamp
