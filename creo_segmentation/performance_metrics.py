# Description: This file contains the implementation of the performance metrics used to evaluate the performance of the model.

# TODO: make sure to not include the absent cases (ground truth number = 0) in performance calculations

def preprocess_cf_list(cf_list):
    [true_positives, true_negatives, false_positives, false_negatives] = cf_list
    if true_positives == 0:
        true_positives = 1
    if true_negatives == 0:
        true_negatives = 1
    if false_positives == 0:
        false_positives = 1
    if false_negatives == 0:
        false_negatives = 1
    return true_positives, true_negatives, false_positives, false_negatives

def naive_accuracy(cf_list): # confusion matrix list input
    true_positives, _, _, _ = preprocess_cf_list(cf_list)
    return true_positives # TODO: divide by total number of pixels and multiply by 100 to get percentage

def precision(cf_list):
    true_positives, _, false_positives, _ = preprocess_cf_list(cf_list)
    return true_positives / (true_positives + false_positives)

def recall(cf_list):
    true_positives, _, _, false_negatives = preprocess_cf_list(cf_list)
    return true_positives / (true_positives + false_negatives)

def f1_score(cf_list):
    precision_ = precision(cf_list=cf_list)
    recall_ = recall(cf_list=cf_list)
    return 2 * (precision_ * recall_) / (precision_ + recall_)

def iou(cf_list):
    true_positives, _, false_positives, false_negatives = preprocess_cf_list(cf_list)
    return true_positives / (true_positives + false_positives + false_negatives)

def accuracy(cf_list):
    true_positives, true_negatives, false_positives, false_negatives = preprocess_cf_list(cf_list)
    return (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)