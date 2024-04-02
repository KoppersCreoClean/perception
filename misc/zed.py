import sys
import cv2
import math
import pyzed.sl as sl
import argparse

def get_depth(x, y, point_cloud):
    point_cloud_value = point_cloud.get_value(x, y)
    
    for value in point_cloud_value:
        if value == math.nan:
            print("bogus")
            return 0

    
    print(point_cloud_value, type(point_cloud_value))
    # distance = math.sqrt(point_cloud_value[0]*point_cloud_value[0] + point_cloud_value[1]*point_cloud_value[1] + point_cloud_value[2]*point_cloud_value[2])
    # print(f"Distance to Camera at {{{x};{y}}}: {distance}")
    
    return 0#distance
    
    


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : "+repr(err)+". Exit program.")
        exit()

    # Capture 50 frames and stop
    i = 0
    image = sl.Mat()
    depth = sl.Mat()
    depth_map = sl.Mat()    
    point_cloud = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    while i < 1:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # cv2.imwrite(str(i)+"_L.jpg", image.get_data())
            zed.retrieve_image(image, sl.VIEW.RIGHT)
            # cv2.imwrite(str(i)+"_R.jpg", image.get_data())
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # Retrieve depth Mat. Depth is aligned on the left image
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            image_cv = image.get_data()
            image_depth = depth.get_data()
            # cv2.imwrite(str(i)+".jpg", image_cv)
            # cv2.imwrite(str(i)+"_depth.jpg", image.get_data())
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
            print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
                  timestamp.get_milliseconds()))
            
            print(image.get_width(), image.get_height())
            
            for x in range(0, image.get_width(), 200):
                for y in range(0, image.get_height(), 200):
                    
                    print(f"Depth at {x}, {y}: {get_depth(x, y, point_cloud)}")
                    # depth_map.set_value(x, y, depth(x, y, point_cloud)) 
            # exit()
            cv2.imwrite(str(i)+"_depth_map.jpg", depth_map.get_data())
            i += 1

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()