#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import numpy as np
from gaussian_color import *
from train_color import *
from util import *
from sklearn import metrics

# Highway sequences configuration, range 1050 - 1350
highway_path_in = "/imatge/froldan/work/highway/input/"
highway_path_gt = "/imatge/froldan/work/highway/groundtruth/"

# Fall sequences configuration, range 1460 - 1560
fall_path_in = "/imatge/froldan/work/fall/input/"
fall_path_gt = "/imatge/froldan/work/fall/groundtruth/"

# Traffic sequences configuration, range 950 - 1050
traffic_path_in = "/imatge/froldan/work/traffic/input/"
traffic_path_gt = "/imatge/froldan/work/traffic/groundtruth/"

# Group sequences
path_tests = [highway_path_in,fall_path_in,traffic_path_in]
path_gts = [highway_path_gt,fall_path_gt,traffic_path_gt]
first_frames = [1050,1460,950]
midle_frames = [1199,1509,999]
last_frames = [1349,1559,1049]

# Define color spaces ['RGB','HSV','YCrCb']
colorSpaces=['YCrCb', 'YCrCb', 'RGB']

# Thresholds on gaussian for each experiment:
alphas = [0.5, 1, 0.5]

dataset = [0, 1, 2]
# Connectivity to fill holes [4, 8]
connectivity = '4'
# Pixels of area filltering
minAreaPixels = [10, 20, 40, 60, 80, 100, 120, 140, 160]

# Define accumulators
FP = np.zeros((3,len(minAreaPixels)), np.int)
FN = np.zeros((3,len(minAreaPixels)), np.int)
TP = np.zeros((3,len(minAreaPixels)), np.int)
TN = np.zeros((3,len(minAreaPixels)), np.int)
P = np.zeros((3,len(minAreaPixels)), np.float)
R = np.zeros((3,len(minAreaPixels)), np.float)
F1 = np.zeros((3,len(minAreaPixels)), np.float)
AUC = np.zeros((3,len(minAreaPixels)-1), np.float)

if __name__ == "__main__":

    # Hole filling
    # Post process with hole filling
    # Try different connectivities: 4 and 8
    # Report with AUC & gain for each sequences.
    # Provide qualitative interpretation...
    print("Evaluating model using {} pixels as min area".format(minAreaPixels))
    for cI, colorSpace in enumerate(colorSpaces):
            print("Starting gaussian modelling dataset num: "+str(dataset[cI])+" color space: "+colorSpace+"...")
            alpha = alphas[cI]
            for aI in range(len(minAreaPixels)):
                minAreaP=minAreaPixels[aI]
                mean_matrix, std_matrix = training_color(path_tests[dataset[cI]], first_frames[dataset[cI]], midle_frames[dataset[cI]], alpha, colorSpace);
                FP[dataset[cI],aI], FN[dataset[cI],aI], TP[dataset[cI],aI], TN[dataset[cI],aI], P[dataset[cI],aI], R[dataset[cI],aI], F1[dataset[cI],aI] = gaussian_color(path_tests[dataset[cI]], path_gts[dataset[cI]], midle_frames[dataset[cI]]+1, last_frames[dataset[cI]], mean_matrix, std_matrix, alpha, colorSpace,connectivity, minAreaP)
                print("Computed gaussian modelling dataset num: "+str(dataset[cI])+" color space: "+colorSpace+" with minArea: "+str(minAreaP))
                try:
                    AUC[dataset[cI], aI-1]=metrics.auc(R[dataset[cI], 0:aI+1], P[dataset[cI], 0:aI+1])
                except ValueError:
                    pass
            print("Starting gaussian modelling dataset num: "+str(dataset[cI])+" color space: "+colorSpace+"... done. AUC: "+str(metrics.auc(R[dataset[cI],:],P[dataset[cI],:]))+"\n")

    plt.clf()
    for i in np.arange(P.shape[0]):
        plt.plot(minAreaPixels,F1[i,:],label='F1-Dataset'+str(i)+'_'+colorSpaces[i])
    plt.xlabel('minArea')
    plt.legend()
    plt.savefig('f1_minArea.png')
    plt.clf()
    for i in np.arange(P.shape[0]):
        plt.plot(R[i,:],P[i,:],label='Dataset'+str(i)+'_'+colorSpaces[i])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.legend()
    plt.savefig('prec_rec_minArea.png')

    for i in np.arange(P.shape[0]):
        plt.plot(minAreaPixels[1:],AUC[i,:],label='Dataset'+str(i)+'_'+colorSpaces[i])
    plt.xlabel('minArea')
    plt.ylabel('Precision')
    #plt.legend()
plt.savefig('aucvsPminArea.png')