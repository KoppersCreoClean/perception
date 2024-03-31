import numpy as np
import cv2

class depthEstimator:

    def __init__(self,
                method="disparity", # method to estimate depth ["disparity" / "stereo" / "learned"
                input_source_path=None, # input source object
                stereo_config_file=None, # stereo object
                learned_model_path=None): # learned object
        
        self.method = method
        self.input_source_path = input_source_path
        self.stereo_config_file = stereo_config_file
        self.learned_model_path = learned_model_path

    def estimate_depth(self, left_img, right_img):
        if self.method == "disparity":
            return self.estimate_depth_disparity(left_img, right_img)
        elif self.method == "stereo":
            return self.estimate_depth_stereo(left_img, right_img)
        elif self.method == "learned":
            return self.estimate_depth_learned(left_img, right_img)
        else:
            raise ValueError("Invalid method to estimate depth")        