import numpy as np
import cv2
import skimage
from skimage.filters import threshold_multiotsu
import matplotlib.pyplot as plt
from performance_metrics import *
import configparser

# TODO: export all utility functions to a separate file utils.py

class Resolution:
    width = 1280
    height = 720

class creoSegmenter:

    def __init__(self,
                 operation="evaluation", # operation to perform ["evaluation" / "segmentation" / "get_location"]
                 segmentation_method="bw_thresholding", # method to segment the image
                 creo_threshold=50, # threshold for semi (grey) pixels
                 clean_threshold=130, # threshold for clean (white) pixels
                 kernel_size=5, # kernel size for dilation and erosion [5/3]
                 kernel_iterations=5): # number of iterations for dilation and erosion [5/3/1]
        
        self.operation = operation
        self.segmentation_method = segmentation_method
        self.creo_threshold = creo_threshold
        self.clean_threshold = clean_threshold
        self.kernel_size = kernel_size
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        self.kernel_iterations = kernel_iterations
        self.calibration_file = "zed_calibration.conf"
        
    def init_calibration(self, calibration_file, image_size=Resolution()) :
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

    # Dilation of image 
    def dilate_image(self, image, kernel=None, iterations=None):
        if kernel is None:
            kernel = self.kernel
        if iterations is None:
            iterations = self.kernel_iterations
        return cv2.dilate(image, kernel=kernel, iterations=iterations)

    # Erosion of image 
    def erode_image(self, image, kernel=None, iterations=None):
        if kernel is None:
            kernel = self.kernel
        if iterations is None:
            iterations = self.kernel_iterations
        return cv2.erode(image, kernel=kernel, iterations=iterations)
    
    # Preprocess the image/mask
    def preprocess_image(self, img, x = 401, y = 8, w = 620, h = 710, if_rectify=False, if_mask=False, if_crop=False, if_resize=True, camera='left'):

        original_img = img
        # cv2.imshow("Original Image", original_img)
        if if_rectify: # de-fisheyeing and rectification
            camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = self.init_calibration(self.calibration_file)
            camera_matrix = None
            left_right_image = np.split(img, 2, axis=1)
            if camera == 'left':
                img = cv2.remap(left_right_image[0], map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
                camera_matrix = camera_matrix_left
            else:
                img = cv2.remap(left_right_image[1], map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)
                camera_matrix = camera_matrix_right
        
        if len(img.shape) == 3 and if_mask == False:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # crop is tuned for this specific dataset and camera position
        if if_crop: # crop the image to the region of interest on the pan
            img = img[y:y+h, x:x+w]
        
        if not if_mask: # tuned operations for the images
            img = self.erode_image(img, kernel=np.ones((7, 7), np.uint8), iterations=7)
            img = self.dilate_image(img)
        
        if if_resize: # resize the image to a specific size for easier processing
            img = cv2.resize(img, (450, 750))
            
        return img
    
    # find true positives, false positives, and false negatives in two binary images
    def segmentation_confusion_matrix(self, bin1, bin2):
        true_positives = np.sum(np.logical_and(bin1 == 255, bin2 == 255))
        true_negatives = np.sum(np.logical_and(bin1 == 0, bin2 == 0))
        false_positives = np.sum(np.logical_and(bin1 == 255, bin2 == 0))
        false_negatives = np.sum(np.logical_and(bin1 == 0, bin2 == 255))
        return [true_positives, true_negatives, false_positives, false_negatives]
    
    # create binary masks for creo (red), semi (green), and clean (blue) regions for evaluation
    def get_masks(self, mask):
        # apply thresholding to the channels to get the binary masks
        red_channel = mask[:, :, 2]
        _, binary_creo_mask = cv2.threshold(red_channel, 250, 255, cv2.THRESH_BINARY)
        green_channel = mask[:, :, 1]
        _, binary_semi_mask = cv2.threshold(green_channel, 250, 255, cv2.THRESH_BINARY)
        blue_channel = mask[:, :, 0]        
        _, binary_clean_mask = cv2.threshold(blue_channel, 250, 255, cv2.THRESH_BINARY)
        return binary_creo_mask, binary_semi_mask, binary_clean_mask                
        
    # apply the bw thresholding to the image to get the different regions
    def bw_thresholding_image(self, image, visualize=False):
        creo_region = cv2.inRange(image, 0, self.creo_threshold)
        semi_creo_region = cv2.inRange(image, self.creo_threshold, self.clean_threshold)
        clean_region = cv2.inRange(image, self.clean_threshold, 255)
        if visualize:
            cv2.imshow("Creo Region", creo_region)
            cv2.imshow("Semi Creo Region", semi_creo_region)
            cv2.imshow("Clean Region", clean_region)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return creo_region, semi_creo_region, clean_region
    
    # apply the adaptive bw thresholding to the image to get the different regions
    def bw_adaptive_thresholding_image(self, image, block_size=11, constant=5, visualize=False):
        creo_region = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant)
        creo_region = self.dilate_image(creo_region, kernel=np.ones((5, 5), np.uint8), iterations=2) # tuned for this dataset
        if visualize:
            cv2.imshow("Creo Region", creo_region)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return creo_region
    
    def adaptive_thresholdGaussian(self, img, block_size=7, c=100):
    # Check that the block size is odd and nonnegative
        assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"
        
        # Calculate the local threshold for each pixel using a Gaussian filter
        threshold_matrix = cv2.GaussianBlur(img, (block_size, block_size), 0)
        threshold_matrix = threshold_matrix - c
        
        # Apply the threshold to the input image
        binary = np.zeros_like(img, dtype=np.uint8)
        binary[img >= threshold_matrix] = 255
        
        return binary
    
    def adaptive_thresholdMean(self, img, block_size=31, creo_offset=5, semi_offset=50):
    # Check that the block size is odd and nonnegative
        assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"

        # Calculate the local threshold for each pixel
        height, width = img.shape
        binary = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                # Calculate the local threshold using a square neighborhood centered at (i, j)
                x_min = max(0, i - block_size // 2)
                y_min = max(0, j - block_size // 2)
                x_max = min(height - 1, i + block_size // 2)
                y_max = min(width - 1, j + block_size // 2)
                block = img[x_min:x_max+1, y_min:y_max+1]
                thresh_creo = np.mean(block) + creo_offset
                thresh_semi = np.mean(block) + semi_offset
                if img[i, j] <= thresh_creo:
                    binary[i, j] = 0
                elif img[i, j] > thresh_creo and img[i, j] <= thresh_semi:
                    binary[i, j] = 127
                else:
                    binary[i, j] = 255

        return binary
    
    def gamma_correction(self, img, gamma=5.0):
        ## [changing-contrast-brightness-gamma-correction]
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        res = cv2.LUT(img, lookUpTable)
        ## [changing-contrast-brightness-gamma-correction]

        return res
    
    def clahe_transform(self, img, clipLimit=2.0, tileGridSize=(8, 8)):
        # Apply some preprocessing like histogram equalization, gamma correction, etc.
        # For example, you can try histogram equalization:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        return clahe.apply(img)
    
    def contrast_stretching(self, img, alpha=1.5, beta=0):
        # out = cv2.addWeighted( img, contrast, img, 0, brightness)
        # output = cv2.addWeighted
        output = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return output    
    
    def remove_shadows(self, img):
        # cv2.imshow("Original Image", img)
        rgb_planes = cv2.split(img)
        # fgbg = cv2.createBackgroundSubtractorMOG2(128,cv2.THRESH_BINARY,1)
        # masked_image = fgbg.apply(img)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            # dilated_img = self.dilate_image(plane, kernel=np.ones((5, 5), np.uint8), iterations=2)
            bg_img = cv2.medianBlur(img, 19)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        delta_intensity = np.mean(img) - np.mean(result_norm)
        
        return result, result_norm, delta_intensity
       
    def multiclass_mean_adaptive_threshold(self, image, block_size=21, C=10):
        # Convert image to grayscale if not already
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Initialize the output image
        output_image = np.zeros_like(image)
        
        # Calculate mean adaptive threshold for each class
        for i in range(3):
            class_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                                    cv2.THRESH_BINARY, block_size, C * (i+1))
            # Update the output image by assigning class values based on the thresholded pixels
            output_image[class_threshold == 255] = (i+1) * 255 // 3
            
        return output_image 
    
    def multi_otsu_thresholding(self, image, if_get_loc=False, visualize=False):
        # original_image = image
        _, image_, delta_intensity = self.remove_shadows(image)
        # if if_get_loc:
        #     image = image_

        # print("Delta Intensity: ", delta_intensity)
        # Compute the Otsu thresholds        
        thresholds = skimage.filters.threshold_multiotsu(image, classes=3)
        # print("Thresholds: ", thresholds)
        # thresholds -= int(delta_intensity/6) # testbed images
        # thresholds -= int(delta_intensity) # factory images


        self.creo_threshold = 100
        self.clean_threshold = 170
        # self.creo_threshold = int(thresholds[0])
        # self.clean_threshold = int(thresholds[1]) # testbed images
        
        # image = self.gamma_correction(image, gamma=0.2) # testbed images
        image = self.gamma_correction(image, gamma=0.7) # factory images
        # image = self.clahe_transform(image, clipLimit=2.0, tileGridSize=(8, 8))# testbed images
        # image = self.contrast_stretching(image, alpha=1.5, beta=0) # testbed images
        image = self.clahe_transform(image, clipLimit=4.0, tileGridSize=(8, 8))# factory images
        # image = self.contrast_stretching(image, alpha=0.5, beta=10) # factory images
        # cv2.imshow("Original Image", original_image)
        # cv2.imshow("Gamma Transformed Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()
        
        # bin = self.adaptive_thresholdGaussian(image)
        # bin = self.adaptive_thresholdMean(image)
        # bin = self.multiclass_mean_adaptive_threshold(image)
        # bin = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=21, C=10)
        
        creo_region, semi_creo_region, clean_region = self.bw_thresholding_image(image)
        
        if visualize:
            cv2.imshow("Original Image", image)
            cv2.imshow("Creo Region", creo_region)
            cv2.imshow("Semi Creo Region", semi_creo_region)
            cv2.imshow("Clean Region", clean_region)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return creo_region, semi_creo_region, clean_region
    
    # apply the otus thresholding to the imaqe
    def otsu_thresholding(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded

    # TODO: implement the segmentation methods
    def hsv_segmentation(self, image):
        pass
    def lab_segmentation(self, image):
        pass
    def lbp_segmentation(self, image):
        pass

    # fill the contours in the image
    def fill_contours(self, img):
        img = self.dilate_image(img, kernel=np.ones((5, 5), np.uint8), iterations=7)
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(img,[c], 0, (255,255,255), -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
        return opening

    # get location of the top/left most pixel of the creo region and translate to real world coordinates
    def get_creo_location(self, image, intrinsics, lambda_=0.6): # no intrinsic matrix because we need coords in camera frame
        
        image = self.preprocess_image(image)
        creo_region, semi_creo_region, _ = self.multi_otsu_thresholding(image, if_get_loc=True)
        
        # bitwise and operation to get the creo region
        total_creo_region = cv2.bitwise_or(creo_region, semi_creo_region)
        total_creo_region = self.fill_contours(total_creo_region)
                
        height, width = total_creo_region.shape[:2]
        for y in range(height - 1, -1, -1):  # Iterate over rows in reverse order
            for x in range(width - 1, -1, -1):  # Iterate over columns in reverse order
                if total_creo_region[y, x] == 255:  # Check if pixel value is white (255)
                    creo_location = (x, y)
                    break
            else:
                continue
            break
        
        # conversion of pixel location to real world coordinates
        creo_location_in_camera_frame = [
            lambda_ * (creo_location[0] - intrinsics["cy"]) / intrinsics["fy"],
            lambda_ * (creo_location[1] - intrinsics["cx"]) / intrinsics["fx"],
            lambda_
        ]
        
        return creo_location_in_camera_frame

    # evaluate the performance of the segmentation
    def evaluate_segmentation(self, image_dir, mask_dir, number_of_images, evaluation_method="accuracy", visualize=True):
        # visualize = True
        avg_metric_values = {"creo": 0, "semi": 0, "clean": 0}
        
        switcher = {
            "naive_accuracy": naive_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "iou": iou,
            "accuracy": accuracy
        }
        metric_function = switcher.get(evaluation_method, "Invalid metric")
        # TODO: switcher for segegmentation methods
        
        count_creo = 0
        count_semi = 0
        count_clean = 0
        
        if False: # testbed images
            i = 14
            number_of_images = 20
        # else: # factory images
        #     i = 3# 7, 6 
        #     number_of_images = 4
        else: # factory images
            i = 9
            number_of_images = 14
        
        while i <= number_of_images:

            if i in [14, 15, 16, 17, 18, 19, 20]:
                image = cv2.imread(image_dir + "/test" + str(i) + ".png")
                mask = cv2.imread(mask_dir + "/test" + str(i) + ".png")
            else:
                image = cv2.imread(image_dir + "/test" + str(i) + ".jpg")
                mask = cv2.imread(mask_dir + "/test" + str(i) + ".jpg")
            
            image = self.preprocess_image(image)
            mask = self.preprocess_image(mask, if_mask=True)
            
            creo_mask, semi_mask, clean_mask = self.get_masks(mask)
            if visualize:
                cv2.imshow("Creo Mask", creo_mask)
                cv2.imshow("Semi Mask", semi_mask)
                cv2.imshow("Clean Mask", clean_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            creo_region, semi_creo_region, clean_region = self.multi_otsu_thresholding(image)
            if visualize:
                cv2.imshow("Creo Region", creo_region)
                cv2.imshow("Semi Creo Region", semi_creo_region)
                cv2.imshow("Clean Region", clean_region)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # get the confusion matrix for the different regions
            creo_metrics = self.segmentation_confusion_matrix(creo_mask, creo_region)
            semi_metrics = self.segmentation_confusion_matrix(semi_mask, semi_creo_region)
            clean_metrics = self.segmentation_confusion_matrix(clean_mask, clean_region)
            if visualize:
                print("Creo Metrics: ", creo_metrics)
                print("Semi Metrics: ", semi_metrics)
                print("Clean Metrics: ", clean_metrics)

            # calculate and add the metric value for each region
            avg_metric_values["creo"] += metric_function(creo_metrics)
            if metric_function(creo_metrics) != 0:
                count_creo += 1
            avg_metric_values["semi"] += metric_function(semi_metrics)
            if metric_function(semi_metrics) != 0:
                count_clean += 1
            avg_metric_values["clean"] += metric_function(clean_metrics)
            if metric_function(clean_metrics) != 0:
                count_semi += 1

            i += 1

        # to avoid division by zero
        if count_clean == 0: count_clean = 1
        if count_semi == 0: count_semi = 1
        if count_creo == 0: count_creo = 1

        # calculate the average metric value for each region
        avg_metric_values["creo"] /= count_creo
        avg_metric_values["semi"] /= count_semi
        avg_metric_values["clean"] /= count_clean

        print("Average "+evaluation_method+" for Creo Region: ", avg_metric_values["creo"])
        print("Average "+evaluation_method+" for Semi Region: ", avg_metric_values["semi"])
        print("Average "+evaluation_method+" for Clean Region: ", avg_metric_values["clean"])
        
        return avg_metric_values

    # get the segmented regions of the image
    def get_segmentations(self, image, segmentation_method="multi_otsu_thresholding", visualize=True):
        self.segmentation_method = segmentation_method
        image = self.preprocess_image(image)
        if self.segmentation_method == "bw_thresholding":
            return self.bw_thresholding_image(image, visualize=visualize)
        elif self.segmentation_method == "bw_adaptive_thresholding":
            return self.bw_adaptive_thresholding_image(image, visualize=visualize)
        elif self.segmentation_method == "multi_otsu_thresholding":
            return self.multi_otsu_thresholding(image, visualize=visualize)
        elif self.segmentation_method == "otu_thresholding":
            return self.otsu_thresholding(image)
        elif self.segmentation_method == "hsv_segmentation":
            return self.hsv_segmentation(image)
        elif self.segmentation_method == "lab_segmentation":
            return self.lab_segmentation(image)
        else:
            print("Invalid segmentation method")
            return None