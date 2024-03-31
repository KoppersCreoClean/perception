import cv2
import pyzed.sl as sl
from creo_segment import creoSegmenter
import argparse

def zed_cap_image():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 30  # Set fps at 30

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

def main():

    # TODO: Consolidate as a ros node

    segmenter = creoSegmenter()

    # evaluating the segmentation model    
    segmenter.evaluate_segmentation("../../data/creo_segmentation/images",
                                    "../../data/creo_segmentation/masks",
                                    13, 
                                    evaluation_method="accuracy", 
                                    visualize=False)
    
    # running the creo segmentation model on real-time camera feed
    image, _ = zed_cap_image()
    segmenter.get_segmentations(image)

    # running the creo segmentation model on a single image to get the creo location
    intrinsics_zedL = {
        "fx": 1.3995,
        "fy": 1.3995,
        "cx": 1.16865,
        "cy": 0.64013
    }
    location = segmenter.get_creo_location(image, intrinsics_zedL)
    print(location)

if __name__ == "__main__":
    main()