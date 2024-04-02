import cv2
import numpy as np

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

def otsu_threshold(image):
        # Convert the image to grayscale if it's not already
        if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresholded

def segment_colors(image):
        # Split the image into its color channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        y, cr, cb = cv2.split(image)

        # Apply Otsu's thresholding to each channel
        thresholded_b = otsu_threshold(y)
        thresholded_g = otsu_threshold(cr)
        thresholded_r = otsu_threshold(cb)

        # Combine the thresholds to get the segmented image
        segmented_image = cv2.merge([thresholded_b, thresholded_g, thresholded_r])
        # segmented_image = thresholded_r

        return segmented_image

def multi_otsu_threshold(image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (600, 1000))
        image = erode_image(image)
        # Get the segmented image
        segmented_image = segment_colors(image)
        # Save the Lab image as JPEG
        cv2.imwrite("./data/interim.jpg", segmented_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        #  Display the original and segmented images
        # cv2.imshow('Original Image', image)
        # cv2.imshow('Segmented Image', segmented_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # orig_img = segmented_image
        # blue : clean
        # pink + yellow + green + white : semi
        # black : creo
        
        gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
        
        # Define color thresholds for red, cyan, pink, green, and white
        red_lower = np.array([0, 0, 100], dtype=np.uint8)
        red_upper = np.array([100, 100, 255], dtype=np.uint8)

        cyan_pink_green_white_lower = np.array([0, 128, 0], dtype=np.uint8)
        cyan_pink_green_white_upper = np.array([255, 255, 255], dtype=np.uint8)

        black_lower = np.array([0, 0, 0], dtype=np.uint8)
        black_upper = np.array([20, 20, 20], dtype=np.uint8)

        # Create masks using inRange
        red_mask = cv2.inRange(segmented_image, red_lower, red_upper)
        # cv2.imshow("Red Mask", red_mask)
        
        cyan_pink_green_white_mask = cv2.inRange(segmented_image, cyan_pink_green_white_lower, cyan_pink_green_white_upper)
        black_mask = cv2.inRange(segmented_image, black_lower, black_upper)
        
        # cv2.imshow("Cyan, Pink, Green, White Mask", cyan_pink_green_white_mask)
        # cv2.imshow("Black Mask", black_mask)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # exit()

        # Convert red pixels to white
        segmented_image[red_mask > 0] = [255, 255, 255]

        # Convert cyan, pink, green, white pixels to grey
        segmented_image[cyan_pink_green_white_mask > 0] = [128, 128, 128]

        # Convert black pixels to black (no change)
        segmented_image[black_mask > 0] = [0, 0, 0]

        # Display the original and processed images
        # cv2.imshow("Original Image", orig_img)#cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        cv2.imshow("Processed Image", segmented_image)#cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2RGB))

        # # Display the original and segmented images
        # cv2.imshow('Original Image', image)
        # cv2.imshow('Segmented Image', segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def apply_otsu_segmentation(image_path):
        # Load the image
        image = cv2.imread(image_path, 0)

        print(image.shape)
        image = cv2.resize(image, (600, 1000))

        image = erode_image(image)

        # Apply Otsu's thresholding
        _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # _, segmented_image = cv2.threshold(image, 0, 255, cv2.ADAPTIVE_THRESH_MEAN_C + cv2.THRESH_OTSU)
        print(segmented_image.shape)

        # Display the segmented image
        cv2.imshow("Segmented Image", segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def rgb_to_lab_to_rgb(image_path):
        # Read the RGB image
        rgb_image = cv2.imread(image_path)
        rgb_image = cv2.resize(rgb_image, (600, 1000))

        # Convert RGB to Lab
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2Lab)

        # Extract L, a, and b channels
        L, a, b = cv2.split(lab_image)

        # Do some processing on the Lab channels (for example, invert the L channel)
        # In this example, we'll keep L as it is, and set a and b to constant values
        a[:] = 128
        b[:] = 128

        # Merge the channels back to Lab
        modified_lab_image = cv2.merge([L, a, b])

        # Convert Lab back to RGB
        modified_rgb_image = cv2.cvtColor(modified_lab_image, cv2.COLOR_Lab2BGR)

        # Display the original and modified images
        cv2.imshow("Original RGB Image", rgb_image)
        cv2.imshow("Modified Lab to RGB Image", modified_rgb_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Replace 'your_image_path.jpg' with the path to your actual image file
# image_path = './data/primary/images/test2.jpg'
# image_path = "./primary/images/test2.jpg"
# rgb_to_lab_to_rgb(image_path)

if __name__ == "__main__":    
        #apply_otsu_segmentation("./data/test1.jpg")
        i = 1
        while i <= 13:
                multi_otsu_threshold(f"./primary/images/test{i}.jpg")
                i += 1