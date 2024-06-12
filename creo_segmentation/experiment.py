import os
import cv2
import numpy as np

# improt Kmeans clustering from skimage
from sklearn.cluster import KMeans

SEGMENTATION_THRESHOLD = 50 # threshold for segmentation
BLUR_KERNEL = 3 # Gaussian blur kernel size
ERODE_KERNEL = 3 # erosion kernel size
EROSION_ITERATIONS = 7 # number of iterations for erosion #
ALPHA = 2.5 # contrast stretching exponent - gamma correction
BETA = 0 # contrast stretching bias - gamma correction
CLIP_LIMIT = 2 # CLAHE clip limit

def KMeans_clustering(img, k=3):
    """KMeans clustering for image segmentation"""
    
    # Reshape the image to 2D array of pixels
    img = img.reshape((-1, 1))
    
    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, algorithm='elkan')
    
    # Fit the model to the image
    kmeans.fit(img)
    
    # Get the labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # sort the centers in ascending order
    centers = np.sort(centers, axis=0)
    
    print(centers, centers.shape)
    
    # Reshape the labels to the original image shape
    labels = labels.reshape(img.shape)
    
    # Determine the threshold for segmentation
    threshold_creo = (centers[0] + centers[1])/2
    threshold_semi = (centers[1] + centers[2])/2
    print(threshold_creo, threshold_semi)
    # (centers[0]*np.sum(labels == 0) + centers[1]*np.sum(labels == 1))/(np.sum(labels == 0) + np.sum(labels == 1))
    
    return threshold_semi.astype(int)

def preprocess(
                img, 
                visualize=True, # visualize the preprocessing steps
                clipLimit=CLIP_LIMIT, # CLAHE clip limit
                alpha=ALPHA, # contrast stretching exponent
                beta=BETA, # contrast stretching bias
                blur_kernel=BLUR_KERNEL, # blur kernel size
                erode_kernel=ERODE_KERNEL, # erode kernel size
                erode_iterations=EROSION_ITERATIONS, # number of iterations for erosion 
                segmentation_threshold=SEGMENTATION_THRESHOLD, # threshold for segmentation
                ):
    "Preprocess the image to get the anode and mandrel regions"

    if visualize:
        cv2.imshow("Original Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Contrast Limited Adaptive Histogram Equalization
    # Types of HIstorgram Equalization:
    # 1. Global Histogram Equalization
    img = cv2.equalizeHist(img)
    # 2. Local Histogram Equalization
    
    
    
    # 3. Contrast Limited Adaptive Histogram Equalization
    
    
    clahe = cv2.createCLAHE(clipLimit=clipLimit)
    # img = clahe.apply(img)

    # if visualize:
    #     cv2.imshow("Equalized Image", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # Gaussian blur
    img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)

    # Further erosion to remove noise and small blobs
    kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
    img = cv2.erode(img, kernel, iterations=erode_iterations)

    if visualize:
        cv2.imshow("Preprocessed Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Scale the images pixel values to 0-255 for even application of gamma correction        
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Contrast streching the middle intensity values using gamma correction
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Scale the images pixel values to 0-255 for even application of segmentation threshold
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    if visualize:
        cv2.imshow("Contrast Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  

    # Threshold the images to get anode and mandrel regions
    segmentation_threshold = KMeans_clustering(img)
    print(segmentation_threshold)
    _, img = cv2.threshold(img, int(segmentation_threshold), 255, cv2.THRESH_BINARY)

    # Invert the image to focus on darker regions
    img = cv2.bitwise_not(img)

    if visualize:
        cv2.imshow("Thresholded Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


if __name__ == "__main__":
    
    # read all th eimages in the folder
    for file in os.listdir("./images"):
        # if not file.endswith(".png"):
        #     continue
        print(file)
        img = cv2.imread("./images/"+file, cv2.IMREAD_GRAYSCALE)
        # resize the image
        img = cv2.resize(img, (400, 600))
        preprocess(img)
        # exit()
