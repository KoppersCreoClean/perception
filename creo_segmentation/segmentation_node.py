import cv2
import numpy as np
from creo_segment import creoSegmenter
import configparser
import argparse


def init_calibration(calibration_file, image_size) :

    cameraMarix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array([])

    config = configparser.ConfigParser()
    config.read(calibration_file)

    check_data = True
    resolution_str = ''
    if image_size.width == 2208 :
        resolution_str = '2K'
    elif image_size.width == 1920 :
        resolution_str = 'FHD'
    elif image_size.width == 1280 :
        resolution_str = 'HD'
    elif image_size.width == 672 :
        resolution_str = 'VGA'
    else:
        resolution_str = 'HD'
        check_data = False

    T_ = np.array([-float(config['STEREO']['Baseline'] if 'Baseline' in config['STEREO'] else 0),
                   float(config['STEREO']['TY_'+resolution_str] if 'TY_'+resolution_str in config['STEREO'] else 0),
                   float(config['STEREO']['TZ_'+resolution_str] if 'TZ_'+resolution_str in config['STEREO'] else 0)])


    left_cam_cx = float(config['LEFT_CAM_'+resolution_str]['cx'] if 'cx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_cy = float(config['LEFT_CAM_'+resolution_str]['cy'] if 'cy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fx = float(config['LEFT_CAM_'+resolution_str]['fx'] if 'fx' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_fy = float(config['LEFT_CAM_'+resolution_str]['fy'] if 'fy' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k1 = float(config['LEFT_CAM_'+resolution_str]['k1'] if 'k1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k2 = float(config['LEFT_CAM_'+resolution_str]['k2'] if 'k2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p1 = float(config['LEFT_CAM_'+resolution_str]['p1'] if 'p1' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p2 = float(config['LEFT_CAM_'+resolution_str]['p2'] if 'p2' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_p3 = float(config['LEFT_CAM_'+resolution_str]['p3'] if 'p3' in config['LEFT_CAM_'+resolution_str] else 0)
    left_cam_k3 = float(config['LEFT_CAM_'+resolution_str]['k3'] if 'k3' in config['LEFT_CAM_'+resolution_str] else 0)


    right_cam_cx = float(config['RIGHT_CAM_'+resolution_str]['cx'] if 'cx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_cy = float(config['RIGHT_CAM_'+resolution_str]['cy'] if 'cy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fx = float(config['RIGHT_CAM_'+resolution_str]['fx'] if 'fx' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_fy = float(config['RIGHT_CAM_'+resolution_str]['fy'] if 'fy' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k1 = float(config['RIGHT_CAM_'+resolution_str]['k1'] if 'k1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k2 = float(config['RIGHT_CAM_'+resolution_str]['k2'] if 'k2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p1 = float(config['RIGHT_CAM_'+resolution_str]['p1'] if 'p1' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p2 = float(config['RIGHT_CAM_'+resolution_str]['p2'] if 'p2' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_p3 = float(config['RIGHT_CAM_'+resolution_str]['p3'] if 'p3' in config['RIGHT_CAM_'+resolution_str] else 0)
    right_cam_k3 = float(config['RIGHT_CAM_'+resolution_str]['k3'] if 'k3' in config['RIGHT_CAM_'+resolution_str] else 0)

    R_zed = np.array([float(config['STEREO']['RX_'+resolution_str] if 'RX_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['CV_'+resolution_str] if 'CV_' + resolution_str in config['STEREO'] else 0),
                      float(config['STEREO']['RZ_'+resolution_str] if 'RZ_' + resolution_str in config['STEREO'] else 0)])

    R, _ = cv2.Rodrigues(R_zed)
    cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                         [0, left_cam_fy, left_cam_cy],
                         [0, 0, 1]])

    cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                          [0, right_cam_fy, right_cam_cy],
                          [0, 0, 1]])

    distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1], [left_cam_p2], [left_cam_k3]])

    distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2], [right_cam_p1], [right_cam_p2], [right_cam_k3]])

    T = np.array([[T_[0]], [T_[1]], [T_[2]]])
    R1 = R2 = P1 = P2 = np.array([])

    R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                       cameraMatrix2=cameraMatrix_right,
                                       distCoeffs1=distCoeffs_left,
                                       distCoeffs2=distCoeffs_right,
                                       R=R, T=T,
                                       flags=cv2.CALIB_ZERO_DISPARITY,
                                       alpha=0,
                                       imageSize=(image_size.width, image_size.height),
                                       newImageSize=(image_size.width, image_size.height))[0:4]

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (image_size.width, image_size.height), cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (image_size.width, image_size.height), cv2.CV_32FC1)

    cameraMatrix_left = P1
    cameraMatrix_right = P2

    return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y

class Resolution :
    width = 1280
    height = 720

# def main() :
#     # Open the ZED camera
#     cap = cv2.VideoCapture(4)
#     if cap.isOpened() == 0:
#         exit(-1)

#     image_size = Resolution()
#     image_size.width = 1280
#     image_size.height = 720

    # # Set the video resolution to HD720
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width*2)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)

    # calibration_file = "zed_calibration.conf"

    # camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(calibration_file, image_size)

    # while True :
    #     # Get a new frame from camera
    #     retval, frame = cap.read()
    #     # Extract left and right images from side-by-side
    #     left_right_image = np.split(frame, 2, axis=1)
    #     # Display images

    #     left_rect = cv2.remap(left_right_image[0], map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
    #     right_rect = cv2.remap(left_right_image[1], map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)

    #     cv2.imshow("left RECT", left_rect)
    #     cv2.imshow("right RECT", right_rect)
    #     if cv2.waitKey(30) >= 0 :
    #         break

    # exit(0)


def main():

    # TODO: Consolidate as a ros node

    segmenter = creoSegmenter()
    
    image_size = Resolution()
    
    intrinsics_zedL = {
        "fx": 1.3995,
        "fy": 1.3995,
        "cx": 1.16865,
        "cy": 0.64013
    }
    
    calibration_file = "zed_calibration.conf"

    # camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(calibration_file, image_size)

    # evaluating the segmentation model    
    # segmenter.evaluate_segmentation("../../data/creo_segmentation/images",
    #                                 "../../data/creo_segmentation/masks",
    #                                 2,
    #                                 evaluation_method="accuracy", 
    #                                 visualize=False)
    
    # running the creo segmentation model on real-time camera feed
    # image, _ = zed_cap_image()
    # for i in range(0, 6):
    #     # print(i)
    #     image = cv2.imread("../left_" + str(i) + ".jpg")
    #     # cv2.imshow("image", image)
    #     image_size.width, image_size.height = image.shape[1], image.shape[0]
    #     camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = init_calibration(calibration_file, image_size)
    #     image = cv2.remap(image, map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("rectified image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # segmenter.get_segmentations(image, visualize=True)
        
    # while True :
    #     # Get a new frame from camera
    #     retval, frame = cap.read()
    #     # Extract left and right images from side-by-side
    #     left_right_image = np.split(frame, 2, axis=1)
    #     # Display images

    #     left_rect = cv2.remap(left_right_image[0], map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
    #     right_rect = cv2.remap(left_right_image[1], map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)

    #     cv2.imshow("left RECT", left_rect)
    #     cv2.imshow("right RECT", right_rect)
    #     if cv2.waitKey(30) >= 0 :
    #         break
        
        # segmenter.get_segmentations(image, visualize=True)

    # running the creo segmentation model on a single image to get the creo location
    image = cv2.imread("../left_0.jpg")
    location = segmenter.get_creo_location(image, intrinsics_zedL)
    print(location)

if __name__ == "__main__":
    main()