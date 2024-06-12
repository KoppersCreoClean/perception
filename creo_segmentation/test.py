import cv2
import numpy as np

def get_masks(mask):
    # apply thresholding to the channels to get the binary masks
    red_channel = mask[:, :, 2]
    cv2.imshow('Red Channel', red_channel)
    _, binary_creo_mask = cv2.threshold(red_channel, 250, 255, cv2.THRESH_BINARY)
    green_channel = mask[:, :, 1]
    cv2.imshow('Green Channel', green_channel)
    _, binary_semi_mask = cv2.threshold(green_channel, 250, 255, cv2.THRESH_BINARY)
    blue_channel = mask[:, :, 0]     
    cv2.imshow('Blue Channel', blue_channel)   
    _, binary_clean_mask = cv2.threshold(blue_channel, 250, 255, cv2.THRESH_BINARY)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return binary_creo_mask, binary_semi_mask, binary_clean_mask

# Load the image
image = cv2.imread('./unnamed.png')

# Convert the image to BGR (if it's not already in BGR format)
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Define the lower and upper bounds for red color
lower_red = np.array([200, 0, 0])
upper_red = np.array([255, 100, 100])

# Define the lower and upper bounds for red color
lower_green = np.array([0, 200, 0])
upper_green = np.array([100, 255, 100])

# Define the lower and upper bounds for red color
lower_blue = np.array([0, 0, 200])
upper_blue = np.array([100, 100, 255])

# Create a mask for the red pixels
mask_red = cv2.inRange(image_bgr, lower_red, upper_red)

mask_green = cv2.inRange(image_bgr, lower_green, upper_green)

mask_blue = cv2.inRange(image_bgr, lower_blue, upper_blue)

# Set the red pixels to black
image_bgr[mask_red > 0] = [0, 0, 0]

# Set the green pixels to gold
image_bgr[mask_green > 0] = [255, 230, 0]

# Set the blue pixels to grey
image_bgr[mask_blue > 0] = [200, 200, 200]

# Convert the image back to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Display the result
cv2.imshow('Red Pixels to Black', image_rgb)
cv2.imwrite('unnamed.png', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()