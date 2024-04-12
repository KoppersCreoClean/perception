import cv2
import numpy as np
 
class ZedCapture:
    def __init__(self, 
                camera_resolution=cv2.CAP_PROP_FRAME_WIDTH, 
                camera_fps=cv2.CAP_PROP_FRAME_HEIGHT):

        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened() == 0:
            exit(-1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def zed_cap_image(self, side='left', visualize=True):
        _, frame = self.cap.read()
        left_right_image = np.split(frame, 2, axis=1)
        if side == 'left':
            if visualize:
                cv2.imshow("left", left_right_image[0])
            image = left_right_image[0]
        elif side == 'right':
            if visualize:
                cv2.imshow("right", left_right_image[1])
            image = left_right_image[1]
        elif side == 'both':
            if visualize:
                cv2.imshow("frame", frame)
            image = frame
        self.close()
        return image
        
    def zed_cap_video(self, side='left', visualize=True):
        while True:
            _, frame = self.cap.read()
            left_right_image = np.split(frame, 2, axis=1)
            if side == 'left':
                if visualize:
                    cv2.imshow("left", left_right_image[0])
                image = left_right_image[0]
            elif side == 'right':
                if visualize:
                    cv2.imshow("right", left_right_image[1])
                image = left_right_image[1]
            elif side == 'both':
                if visualize:
                    cv2.imshow("frame", frame)
                image = frame
            if cv2.waitKey(30) >= 0:
                break
        self.close()
        return image
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()