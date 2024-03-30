import cv2
import numpy as np
import random

# Function to apply Gaussian noise
def add_gaussian_noise(image):
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + gauss
    return noisy

# Function to apply Gaussian blur
def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Function to flip image vertically
def flip_vertical(image):
    return cv2.flip(image, 0)

# Function to flip image horizontally
def flip_horizontal(image):
    return cv2.flip(image, 1)

# Function to crop and resize image
def crop_and_resize(image, crop_size=(100, 100)):
    h, w = image.shape[:2]
    start_y = random.randint(0, h - crop_size[1])
    start_x = random.randint(0, w - crop_size[0])
    return cv2.resize(image[start_y:start_y + crop_size[1], start_x:start_x + crop_size[0]], (224, 224))

# Example usage
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# Apply augmentations
augmented_image = crop_and_resize(image)
augmented_image = apply_gaussian_blur(augmented_image)
augmented_image = add_gaussian_noise(augmented_image)
augmented_image = flip_vertical(augmented_image)
augmented_image = flip_horizontal(augmented_image)

# Display the augmented image
cv2.imshow('Augmented Image', augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
