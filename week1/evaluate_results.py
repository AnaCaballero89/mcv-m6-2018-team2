import cv2
import os
import numpy as np
from sklearn.metrics import confusion_matrix

results_path = './results/highway'
gt_path = './results/groundtruth'
method = 'A' #A or B
labels = [0, 255]
files_ids = list(range(1201, 1401))
fp = 0
fn = 0
tp = 0
tn = 0


def evaluate_sample(prediction, groundtruth):
    prediction = cv2.imread(prediction, 0).flatten()
    prediction = [pred * 255 for pred in prediction]
    groundtruth = cv2.imread(groundtruth, 0).flatten()
    groundtruth = [0 if x < 255 else x for x in groundtruth]
    #print(list(set(prediction)))
    #print(list(set(groundtruth)))
    conf_mat = confusion_matrix(groundtruth, prediction, labels=labels)
    if len(labels) is 2:
        FP = conf_mat[0, 1]
        FN = conf_mat[1, 0]
        TP = conf_mat[1, 1]
        TN = conf_mat[0, 0]
    else:
        FP = conf_mat.sum(axis=0) - np.diag(conf_mat)
        FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
        TP = np.diag(conf_mat)
        TN = conf_mat.sum() - (FP + FN + TP)
    return FP, FN, TP, TN


for id in files_ids:
    filename = 'test_' + method + '_00' + str(id) + '.png'
    gt_filename = 'gt00' + str(id) + '.png'
    result_file = os.path.join(results_path, filename)
    gt_file = os.path.join(gt_path, gt_filename)
    print('Analyzing image {}'.format(result_file))
    FP, FN, TP, TN = evaluate_sample(result_file, gt_file)
    fp += FP.sum()
    fn += FN.sum()
    tp += TP.sum()
    tn += TN.sum()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
fscore = 2 * precision * recall / (precision + recall)
print('Precision: {}  Recall: {}  F-Score: {}'.format(precision, recall, fscore))