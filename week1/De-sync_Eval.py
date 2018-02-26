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
De_sync=[3,5,10,20]
fp = np.zeros([len(De_sync),len(files_ids)])
fn = np.zeros([len(De_sync),len(files_ids)])
tp = np.zeros([len(De_sync),len(files_ids)])
tn = np.zeros([len(De_sync),len(files_ids)])
Precision=np.zeros([len(De_sync),len(files_ids)])
Recall=np.zeros([len(De_sync),len(files_ids)])
f1=np.zeros([len(De_sync),len(files_ids)])

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


for De in np.arange(len(De_sync)):
    for id in files_ids:
        Delay=De_sync[De]
        filename = 'test_' + method + '_00' + str(id) + '.png'
        gt_filename = 'gt00' + str(id+Delay) + '.png'
        result_file = os.path.join(results_path, filename)
        gt_file = os.path.join(gt_path, gt_filename)
        print('Analyzing image {}'.format(result_file))
        fp[De,id-files_ids[1]], fn[De,id-files_ids[1]], tp[De,id-files_ids[1]], tn[De,id-files_ids[1]] = evaluate_sample(result_file, gt_file)

for De in np.arange(len(De_sync)):
    Precision[De,:]=tp[De,:]/(tp[De,:]+fp[De,:])
    Recall[De,:]=tp[De,:]/(tp[De,:]+fn[De,:])
    f1[De,:]=2*Precision[De,:]*Recall[De,:]/(Precision[De,:]+Recall[De,:])


plt.figure(1)
for De in np.arange(len(De_sync)):
    plt.plot(f1[De,:],label='F1-'+str(De_sync[De])+'framesDelay')

plt.xlabel('frame')
plt.legend()


