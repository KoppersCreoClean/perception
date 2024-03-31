import numpy as np
import cv2
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
        
    # Dilation of image 
    def dilate_image(self, image):
        return cv2.dilate(image, self.kernel, iterations=self.kernel_iterations)

    # Erosion of image 
    def erode_image(self, image):
        return cv2.erode(image, self.kernel, iterations=3)
    
    # preprocess the image/mask
    def preprocess_image(self, img, if_mask=False):
        if len(img.shape) == 3 and if_mask == False:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (600, 1000))
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
    
    # apply the otus thresholding to the image
    def otu_thresholding(self, image):
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

        image = self.preprocess_image(image)
        creo_region, _, _ = self.bw_thresholding_image(image)
        creo_locations = np.where(creo_region == 255)
        creo_location = [creo_locations[0][0], creo_locations[1][0]]
        
        # conversion of pixel location to real world coordinates
        creo_location_in_camera_frame = [
            lambda_ * (creo_location[0] - intrinsics["cy"]) / intrinsics["fy"],
            lambda_ * (creo_location[1] - intrinsics["cx"]) / intrinsics["fx"],
            lambda_
        ]
        
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
            creo_region, semi_creo_region, clean_region = self.bw_thresholding_image(image)
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
    def get_segmentations(self, image):
        image = self.preprocess_image(image)
        if self.segmentation_method == "bw_thresholding":
            return self.bw_thresholding_image(image, visualize=True)
        elif self.segmentation_method == "otu_thresholding":
            return self.otu_thresholding_mask(image)
        elif self.segmentation_method == "hsv_segmentation":
            return self.hsv_segmentation(image)
        elif self.segmentation_method == "lab_segmentation":
            return self.lab_segmentation(image)
        else:
            print("Invalid segmentation method")
            return None