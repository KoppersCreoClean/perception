import numpy as np
import cv2
import skimage
from skimage.filters import threshold_multiotsu
import matplotlib.pyplot as plt
from performance_metrics import *

class creoSegmenter:

    def __init__(self,
                 operation="evaluation", # operation to perform ["evaluation" / "segmentation" / "get_location"]
                 segmentation_method="bw_thresholding", # method to segment the image
                 creo_threshold=50, # threshold for semi (grey) pixels
                 clean_threshold=170, # threshold for clean (white) pixels
                 kernel_size=5, # kernel size for dilation and erosion [5/3]
                 kernel_iterations=3): # number of iterations for dilation and erosion [5/3/1]
        
        self.operation = operation
        self.segmentation_method = segmentation_method
        self.creo_threshold = creo_threshold
        self.clean_threshold = clean_threshold
        self.kernel_size = kernel_size
        self.kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        self.kernel_iterations = kernel_iterations

    def remove_shadows(self, img):    
        rgb_planes = cv2.split(img)

        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            # dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8))
            bg_img = cv2.medianBlur(img, 19)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        delta_intensity = np.mean(img) - np.mean(result_norm)
        
        # print("Delta Intensity: ", delta_intensity)
        # exit()

        return result, result_norm, delta_intensity
        
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
    
    # preprocess the image/mask
    def preprocess_image(self, img, x = 100, y = 50, w = 450, h = 700, if_mask=False, if_crop=True): # crop is tuned for this specific dataset and position
        if len(img.shape) == 3 and if_mask == False:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (600, 1000))
        if if_crop:
            img = img[y:y+h, x:x+w]
        if not if_mask:
            img = self.erode_image(img)
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
    
     # apply the bw thresholding to the image to get the different regions
    
    def bw_adaptive_thresholding_image(self, image, block_size=11, constant=5, visualize=False):
        creo_region = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, constant)
        creo_region = self.dilate_image(creo_region, kernel=np.ones((5, 5), np.uint8), iterations=2) # tuned for this dataset
        if visualize:
            cv2.imshow("Creo Region", creo_region)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return creo_region
    
    def multi_otsu_thresholding(self, image_, visualize=False):
        _, image, delta_intensity = self.remove_shadows(image_)
        # cv2.imshow("Image", image)
        thresholds = skimage.filters.threshold_multiotsu(image, classes=3)
        # thresholds += int(delta_intensity)
        # print("Thresholds: ", thresholds, len(thresholds), type(thresholds), type(thresholds[0]))
        regions = np.digitize(image, bins=thresholds)
        # print("Thresholds: ", thresholds)
        # print("Regions: ", regions, regions.shape, regions.dtype)
        
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 3.5))
        
        creo_region = cv2.inRange(image, 0, int(thresholds[0]))
        semi_creo_region = cv2.inRange(image, int(thresholds[0]), int(thresholds[1])) #+ creo_region
        clean_region = cv2.inRange(image, int(thresholds[1]), 255)
        # cv2.imshow("Creo Region", creo_region)

        if visualize:
            cv2.imshow("Original-Unfiltered Image", image_)
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

    # get location of the top/left most pixel of the creo region and translate to real world coordinates
    def get_creo_location(self, image, intrinsics, lambda_=0.6): # no intrinsic matrix because we need coords in camera frame
        
        cv2.imshow("Original Image", image)
        image = self.preprocess_image(image)
        creo_region, _, _ = self.multi_otsu_thresholding(image)
        # creo_region = 255 - creo_region
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
        dilated = cv2.dilate(creo_region, kernel, iterations=5)
        # _, cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
        # # screenCnt = None

        # # loop over our contours
        # for c in cnts:
        #     # approximate the contour
        #     peri = cv2.arcLength(c, True)
        #     approx = cv2.approxPolyDP(c, 0.3 * peri, True)

        #     cv2.drawContours(creo_region, [cnts[0]], -1, (0, 255, 0), 2)
            
        # exit()
        
        
        # img_canny = cv2.Canny(creo_region, 50, 50)
        # img_dilate = cv2.dilate(img_canny, None, iterations=1)
        # img_erode = cv2.erode(img_dilate, None, iterations=1)

        # mask = np.full(creo_region.shape, 255, "uint8")
        # contours, hierarchies = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # for cnt in contours:
        #     cv2.drawContours(mask, [cnt], -1, 0, -1)
            
        # cv2.imshow("result", dilated)
        # # cv2.waitKey(0)
        # # creo_region = 255 - creo_region
        # cv2.imshow("Creo Region", creo_region)
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # creo_locations = np.where(creo_region == 255)
        # print("Creo Locations: ", creo_locations)
        # print("Creo Locations Shape: ", creo_locations[0].shape)
        # exit()
        height, width = dilated.shape[:2]
        for y in range(height - 1, -1, -1):  # Iterate over rows in reverse order
            for x in range(width - 1, -1, -1):  # Iterate over columns in reverse order
                if creo_region[y, x] == 255:  # Check if pixel value is white (255)
                    creo_location = (x, y)
                    break
            else:
                continue
            break
        # creo_location = [creo_locations[0][0], creo_locations[1][0]]
        
        # conversion of pixel location to real world coordinates
        creo_location_in_camera_frame = [
            lambda_ * (creo_location[0] - intrinsics["cy"]) / intrinsics["fy"],
            lambda_ * (creo_location[1] - intrinsics["cx"]) / intrinsics["fx"],
            lambda_
        ]
        # print("Creo Location in Camera Frame: ", creo_location_in_camera_frame)
        # exit()
        
        return creo_location_in_camera_frame

    # evaluate the performance of the segmentation
    def evaluate_segmentation(self, image_dir, mask_dir, number_of_images, evaluation_method="precision", visualize=False):
        
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
        
        count_creo = 0
        count_semi = 0
        count_clean = 0
        i = 1
        while i <= number_of_images:

            image = cv2.imread(image_dir + "/test" + str(i) + ".jpg")
            mask = cv2.imread(mask_dir + "/test" + str(i) + ".jpg")
            # image = cv2.imread(image_dir + "/left_" + str(i) + ".jpg")
            # mask = cv2.imread(mask_dir + "/left_" + str(i) + ".jpg")
            
            image = self.preprocess_image(image)
            mask = self.preprocess_image(mask, if_mask=True)
            
            creo_mask, semi_mask, clean_mask = self.get_masks(mask)
            # exit()
            if visualize:
                cv2.imshow("Creo Mask", creo_mask)
                cv2.imshow("Semi Mask", semi_mask)
                cv2.imshow("Clean Mask", clean_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            creo_region, semi_creo_region, clean_region = self.multi_otsu_thresholding(image)#self.bw_thresholding_image(image)
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

        if count_clean == 0:
            count_clean = 1
        if count_semi == 0:
            count_semi = 1
        if count_creo == 0:
            count_creo = 1

        # calculate the average metric value for each region
        avg_metric_values["creo"] /= count_creo
        avg_metric_values["semi"] /= count_semi
        avg_metric_values["clean"] /= count_clean

        print("Average "+evaluation_method+" for Creo Region: ", avg_metric_values["creo"])
        print("Average "+evaluation_method+" for Semi Region: ", avg_metric_values["semi"])
        print("Average "+evaluation_method+" for Clean Region: ", avg_metric_values["clean"])

        return avg_metric_values, 

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