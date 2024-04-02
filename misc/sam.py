import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as mask_utils

sam = sam_model_registry["vit_h"](checkpoint="/home/teama24/Desktop/yahuja/CreoClean/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
image = cv2.imread("./data/images/test1.jpg")
masks = mask_generator.generate(image)
# mask = mask_utils.decode(annotation["segmentation"])
# print(masks.shape)
print(type(masks))
print(len(masks))
# print(masks[0].shape)
print(masks)
# cv2.imwrite("mask.jpg", masks)