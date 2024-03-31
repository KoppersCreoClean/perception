# Description: This file contains the implementation of the performance metrics used to evaluate the performance of the model.

def naive_accuracy(true_positives):
    return true_positives

def precision(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)

def recall(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def iou(true_positives, false_positives, false_negatives):
    return true_positives / (true_positives + false_positives + false_negatives)

def accuracy(true_positives, true_negatives, false_positives, false_negatives):
    return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)