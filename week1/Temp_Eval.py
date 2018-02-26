import cv2
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

results_path = './Datasets/results/highway'
gt_path = './Datasets/results/groundtruth'
method = 'A' #A or B
labels = [0, 255]
files_ids = list(range(1201, 1401))
fp = np.zeros(len(files_ids))
fn = np.zeros(len(files_ids))
tp = np.zeros(len(files_ids))
tn = np.zeros(len(files_ids))


def evaluate_sample(prediction, groundtruth):
    prediction = cv2.imread(prediction, 0).flatten()
    prediction = [pred * 255 for pred in prediction]
    groundtruth = cv2.imread(groundtruth, 0).flatten()
    groundtruth = [0 if x < 255 else x for x in groundtruth]
    #print(list(set(prediction)))
    #print(list(set(groundtruth)))
    conf_mat = confusion_matrix(groundtruth, prediction, labels=labels)
    FP = conf_mat[0,1]
    FN = conf_mat[1,0]
    TP = conf_mat[1,1]
    TN = conf_mat[0,0]
    return FP, FN, TP, TN


for id in files_ids:
    filename = 'test_' + method + '_00' + str(id) + '.png'
    gt_filename = 'gt00' + str(id) + '.png'
    result_file = os.path.join(results_path, filename)
    gt_file = os.path.join(gt_path, gt_filename)
    print('Analyzing image {}'.format(result_file))
    fp[id-files_ids[1]], fn[id-files_ids[1]], tp[id-files_ids[1]], tn[id-files_ids[1]] = evaluate_sample(result_file, gt_file)

Precision=tp/(tp+fp)
Recall=tp/(tp+fn)
f1=2*Precision*Recall/(Precision+Recall)

plt.figure(1)
plt.plot(tp,label='True_positive')
plt.plot(tp+fp,label='Total_positive')
plt.xlabel('frame')
plt.legend()

plt.figure(2)
plt.plot(f1,label='F1-score')
plt.xlabel('frame')
plt.legend()
