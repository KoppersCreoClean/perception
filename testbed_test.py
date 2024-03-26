import cv2
import numpy as np

# Dilation of image 
def dilate_image(image):
	kernel = np.ones((3,3), np.uint8)   # 3 or 5
	return cv2.dilate(image, kernel, iterations=3)

# Erosion of image 
def erode_image(image):
	kernel = np.ones((3, 3), np.uint8)   # 3 or 5
	return cv2.erode(image, kernel, iterations=3)# 3 or 1

# Sharpening of image
def sharpen_image(image):
	edges = cv2.Canny(image=image, threshold1=50, threshold2=100) # Canny Edge Detection
	kernel = np.array([[0, -1, 0],
			   [-1, 7,-1],
			   [0, -1, 0]])
	return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

def separate_colors(image_path, mask_path):
    # Load the image

    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (600, 1000))
    image = cv2.imread(image_path)
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (600, 1000))
    image = erode_image(image)
    
    thresholded = image
    # Apply Otsu thresholding
    # _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create masks for black, grey, and white colors
    black_mask = cv2.inRange(thresholded, 0, 50)
    grey_mask = cv2.inRange(thresholded, 51, 169)
    white_mask = cv2.inRange(thresholded, 170, 255)

    # Apply masks to the original image
    black_pixels = cv2.bitwise_and(image, image, mask=black_mask)
    grey_pixels = cv2.bitwise_and(image, image, mask=grey_mask)
    white_pixels = cv2.bitwise_and(image, image, mask=white_mask)
    
    
#     blackMask = cv2.inRange(mask, 0, 50)
    
#     test_segmentation(black_pixels, blackMask)
#     test_segmentation(grey_pixels, greyMask)
#     test_segmentation(white_pixels, whiteMask)

    # Display the separated colors
    cv2.imshow("Original", image)
    cv2.imshow("Black Pixels", black_mask)
    cv2.imshow("Grey Pixels", grey_mask)
    cv2.imshow("White Pixels", white_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_segmentation(img1, img2, str_=""):
    
    # compare the output segmentation against gt mask
    # compare the output segmentation against the ground truth
    
    # Load the image
    # image = cv2.imread("test.jpg")
    # Convert to grayscale
    result = cv2.bitwise_and(img1, img2, mask=None)

    cv2.imshow("Original Image - "+str_, img1)
    cv2.imshow("Mask Image", img2)
    cv2.imshow("Result", result) # intersection of the two images - the true positives
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Count the number of white pixels and black pixels
    white_pixels = np.sum(result == 255)
    # print("white pixels: ", white_pixels)
    black_pixels = np.sum(result == 0)

    # Calculate the total number of pixels
    total_pixels = np.sum(img2 == 255) + 1# 1 added to avoid division by zero
    # print("total pixels in gt: ", total_pixels)
    #img1.size

    # Calculate the percentage of white and black pixels
    percentage_white = (white_pixels / total_pixels) * 100
#     percentage_black = (black_pixels / total_pixels) * 100
    
    return percentage_white
    
#     print("Percentage of white pixels (True Positives):", percentage_white)

# Usage example

def creosote_separation(image_path, mask_path, show_images=False):
        
    mask = cv2.imread(mask_path)
#     print(mask.shape)
    mask = cv2.resize(mask, (300, 500))
    
    # Load the image
    image = cv2.imread(image_path)
#     print(image.shape)
    image = cv2.resize(image, (300, 500))#(600, 1000))
    image = erode_image(image)
    image = dilate_image(image)
    
    # Extract red channel
    red_channel = mask[:, :, 2]
    green_channel = mask[:, :, 1]
    blue_channel = mask[:, :, 0]

    # Thresholding: Convert red pixels to white and others to black
    _, binary_red_channel = cv2.threshold(red_channel, 250, 255, cv2.THRESH_BINARY)
    
    _, binary_green_channel = cv2.threshold(green_channel, 250, 255, cv2.THRESH_BINARY)
    _, binary_blue_channel = cv2.threshold(blue_channel, 250, 255, cv2.THRESH_BINARY)
    
    # Make all blue pixels in binary_blue_channel white
#     binary_red_channel[binary_red_channel > 50] = 255
#     binary_green_channel[binary_green_channel > 50] = 255
#     binary_blue_channel[binary_blue_channel > 50] = 255

    # Create an all-black image
    black_image = np.zeros_like(mask)

    # Merge the red channel back to the black image
    result_image_r = cv2.merge((black_image[:, :, 0], black_image[:, :, 1], binary_red_channel))
    result_image_g = cv2.merge((black_image[:, :, 0], black_image[:, :, 1], binary_green_channel))
    result_image_b = cv2.merge((black_image[:, :, 0], black_image[:, :, 1], binary_blue_channel))
    
    creosote_mask = cv2.cvtColor(result_image_r, cv2.COLOR_BGR2GRAY)
    creosote_mask = cv2.inRange(creosote_mask, 20, 255)
    
    semi_mask = cv2.cvtColor(result_image_g, cv2.COLOR_BGR2GRAY)
    semi_mask = cv2.inRange(semi_mask, 20, 255)
    
    clean_mask = cv2.cvtColor(result_image_b, cv2.COLOR_BGR2GRAY)
    clean_mask = cv2.inRange(clean_mask, 20, 255)
    # -- these are the ground truth masks
    
    # Display the original and result images
    if show_images:
        cv2.imshow("Original Image", image)
        cv2.imshow("creosote Mask", creosote_mask)
        cv2.imshow("semi Mask", semi_mask)
        cv2.imshow("clean Mask", clean_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#**********************************************************************************

#     # Load the image
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (300, 500))#(600, 1000))
#     image = erode_image(image)
    
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (300, 500))
#     image = erode_image(image)
    
    thresholded = image
    # Apply Otsu thresholding
    # _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create masks for black, grey, and white colors, these are the segmentation results
    black_mask = cv2.inRange(thresholded, 0, 70)
    grey_mask = cv2.inRange(thresholded, 71, 130)
    white_mask = cv2.inRange(thresholded, 131, 255)

#     # Apply masks to the original image
#     black_pixels = cv2.bitwise_and(image, image, mask=black_mask)
#     grey_pixels = cv2.bitwise_and(image, image, mask=grey_mask)
#     white_pixels = cv2.bitwise_and(image, image, mask=white_mask)

    # Display the separated colors
    if show_images:
        cv2.imshow("Original", image)
        cv2.imshow("Black Pixels", black_mask)
        cv2.imshow("Grey Pixels", grey_mask)
        cv2.imshow("White Pixels", white_mask)
        # cv2.imshow("Clean Mask", clean_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    acc_creo = test_segmentation(black_mask, creosote_mask, "creosote")
    acc_semi = test_segmentation(grey_mask, semi_mask, "semi")
    acc_clean = test_segmentation(white_mask, clean_mask, "clean")
    
    return np.array([acc_creo, acc_semi, acc_clean])
    
#     separate_colors("./primary_data/images/test2.jpg", "./primary_data/masks/test2.jpg")
    
    
if __name__ == "__main__":

    i = 1
    count_clean = 0
    count_semi = 0
    count_creo = 0
    sum_accuracy = np.zeros(3)
    while i <= 6:
        if i in []:#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            i += 1
            continue
        # print(i)
        image_path = "./data/images/testbed"+str(i)+"-modified.png"
        mask_path = "./data/masks/testbed"+str(i)+".png"
        acc = creosote_separation(image_path, mask_path, show_images=True)
        
        # Count the number of images that have creosote, semi, and clean, to avoid division by zero
        if acc[-1] > 0:
            count_clean += 1
            
        if acc[1] > 0:
            count_semi += 1
            
        if acc[0] > 0:
            count_creo += 1
            
            
        print("accuracy: ", acc)
        sum_accuracy += acc
        i += 1

    sum_accuracy[0] /= (count_creo)
    print(sum_accuracy[0])
    sum_accuracy[1] /= (count_semi)
    print(sum_accuracy[1])
    sum_accuracy[-1] /= (count_clean)
    print(sum_accuracy[-1])
        
    print("mean accuracy: ", sum_accuracy)