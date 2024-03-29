import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import argparse

class accuracyMetrics:
    
    def __init__(self, true_positives, true_negatives, false_positives, false_negatives, metric):
        self.true_positives = true_positives
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.false_negatives = false_negatives
    
    def accuracy(self):
        return (self.true_positives + self.true_negatives) / (self.true_positives + self.true_negatives + self.false_positives + self.false_negatives)
    
    def precision(self):
        return self.true_positives / (self.true_positives + self.false_positives)
    
    def recall(self):
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    def f1_score(self):
        return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
    
    def confusion_matrix(self):
        return np.array([[self.true_positives, self.false_positives], [self.false_negatives, self.true_negatives]])
    
    def naive_accuracy(self):
        return self.true_positives / 100
    

class segmentationTester:
    
    def __init__(self, method="bw_thresholding", input_image=None, ground_truth=None, metric="naive_accuracy"):
        self.method = method
        self.input_image = input_image
        self.ground_truth = ground_truth
        metrics = accuracyMetrics.__init__(self, 0, 0, 0, 0, metric)
        
        # switch case for metric function
        match self.metric_function:
            case "accuracy":
                
         
        if self.method == "bw_thresholding":
            self.bw_thresholding(self)
            
        else:
            raise ValueError("Method not supported")
            
    def bw_thresholding(self):   
                        
        self.true_positives = self.true_positives
        self.true_negatives = self.true_negatives
        self.false_positives = self.false_positives
        self.false_negatives = self.false_negatives

    
    

def main(args):
    
    tester = segmentationTester(args.method, args.input_images, args.ground_truth_images, args.metric)
    
    for idx in range(args.num_images):
        input_image = cv2.imread(args.input_images + str(idx) + ".jpg")
        ground_truth = cv2.imread(args.ground_truth_images + str(idx) + ".jpg")
        
        tester = segmentationTester(args.method, input_image, ground_truth, args.metric)
        
        print("Accuracy: ", tester.accuracy())
        print("Precision: ", tester.precision())
        print("Recall: ", tester.recall())
        print("F1 Score: ", tester.f1_score())
        print("Confusion Matrix: ", tester.confusion_matrix())
        print("Naive Accuracy: ", tester.naive_accuracy())
        
        plt.imshow(input_image)
        plt.show()
        plt.imshow(ground_truth)
        plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_images', type=str, required=True, help='Path to the input images')
    parser.add_argument('--ground_truth_images', type=str, required=True, help='Path to the ground truth images')
    parser.add_argument('--method', type=str, help='Segmentation method to use', default='bw_thresholding')
    parser.add_argument('--num_images', type=int, help='Number of images to process', default=1)
    parser.add_argument('--metric', type=str, help='Metric to use', default='naive_accuracy')
    args = parser.parse_args()
    main(args)