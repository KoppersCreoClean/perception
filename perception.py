import numpy as np
import math
import cv2

# Dilation of image 
def dilate_image(image):
        kernel = np.ones((5,5), np.uint8)   # 3 or 5
        return cv2.dilate(image, kernel, iterations=5)

# Erosion of image 
def erode_image(image):
        kernel = np.ones((5,5), np.uint8)   # 3 or 5
        return cv2.erode(image, kernel, iterations=1)# 3 or 1

# Sharpening of image
def sharpen_image(image):
        edges = cv2.Canny(image=image, threshold1=50, threshold2=100) # Canny Edge Detection
        kernel = np.array([[0, -1, 0],
                           [-1, 7,-1],
                           [0, -1, 0]])
        return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)


# Otsu's thresholding
def otsu_threshold(image):
        # Convert the image to grayscale if it's not already
        if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresholded



def test_segmentation():
    
    # compare the output segmentation against json mask
    # compare the output segmentation against the ground truth
    
    # Load the image
    image = cv2.imread("test.jpg")
    # Convert to grayscale
    pass

def test_depth():
    # compare the output depth against the ground truth
    # compare the output depth against the json mask
    # compare the output depth against the segmentation
    pass

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

def get_segmentation(image):
    # Load the model
    # Load the image
    # Run the model
    # Return the segmentation
    pass

def main():
    # input: image
    
    # Load the image
    
    # Get the segmentation
    
    # Get the depth
    
    # Compare the segmentation against the ground truth
    
    # Compare the depth against the ground truth
    
    # Output conclusions
    
    pass


if __name__ == "__main__":
    main()