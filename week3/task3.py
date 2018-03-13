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
highway_path_in = "datasets/highway/input/"
highway_path_gt = "datasets/highway/groundtruth/"

# Fall sequences configuration, range 1460 - 1560
fall_path_in = "datasets/fall/input/"
fall_path_gt = "datasets/fall/groundtruth/"

# Traffic sequences configuration, range 950 - 1050
traffic_path_in = "datasets/traffic/input/"
traffic_path_gt = "datasets/traffic/groundtruth/"

# Group sequences
path_tests = [highway_path_in,fall_path_in,traffic_path_in]
path_gts = [highway_path_gt,fall_path_gt,traffic_path_gt]
first_frames = [1050,1460,950]
midle_frames = [1199,1509,999]
last_frames = [1349,1559,1049]

# Define color spaces ['RGB','HSV','YCrCb']
colorSpaces=['YCrCb', 'YCrCb', 'RGB']

# Thresholds on gaussian for each experiment:
alphas = np.arange(0.5,3,0.5)
dataset = [0, 1, 2]
# Connectivity to fill holes [4, 8]
connectivity = '8'
# Pixels of area filltering
minAreaPixels = [60, 160]


#Define the morphology
ac_morphology=1; # 1 = apply morphology ; 0 = not to apply morphology
SE1sizes=[3, 7, 11];
SE2sizes=[5, 9, 13];

# Define accumulators
FP = np.zeros((3,len(alphas)), np.int)
FN = np.zeros((3,len(alphas)), np.int)
TP = np.zeros((3,len(alphas)), np.int)
TN = np.zeros((3,len(alphas)), np.int)
P = np.zeros((3,len(alphas)), np.float)
R = np.zeros((3,len(alphas)), np.float)
F1 = np.zeros((3,len(alphas)), np.float)
AUC = np.zeros((3,len(minAreaPixels),len(SE1sizes),len(SE2sizes)), np.float)
Pres = np.zeros((3,len(minAreaPixels),len(SE1sizes),len(SE2sizes),len(alphas)), np.float)
Recall=np.zeros((3,len(minAreaPixels),len(SE1sizes),len(SE2sizes),len(alphas)), np.float)
F1_score=np.zeros((3,len(minAreaPixels),len(SE1sizes),len(SE2sizes),len(alphas)), np.float)
if __name__ == "__main__":

    # Hole filling
    # Post process with hole filling
    # Try different connectivities: 4 and 8
    # Report with AUC & gain for each sequences.
    # Provide qualitative interpretation..
    print("Evaluating model using {} pixels as min area".format(minAreaPixels))
    for cI, colorSpace in enumerate(colorSpaces):
        print("Starting gaussian modelling dataset num: "+str(dataset[cI])+" color space: "+colorSpace+"...")
        alpha = alphas[cI]
        for areaI in range(len(minAreaPixels)):
            for SE1I in range(len(SE1sizes)):
                for SE2I in range(len(SE2sizes)):
                    for aI, alpha in enumerate(alphas):
                        SE1size=SE1sizes[SE1I]
                        SE2size=SE2sizes[SE2I]
                        minAreaP=minAreaPixels[areaI]
                        mean_matrix, std_matrix = training_color(path_tests[dataset[cI]], first_frames[dataset[cI]], midle_frames[dataset[cI]], alpha, colorSpace);
                        FP[dataset[cI],aI], FN[dataset[cI],aI], TP[dataset[cI],aI], TN[dataset[cI],aI], P[dataset[cI],aI], R[dataset[cI],aI], F1[dataset[cI],aI] = gaussian_color(path_tests[dataset[cI]], path_gts[dataset[cI]], midle_frames[dataset[cI]]+1, last_frames[dataset[cI]], mean_matrix, std_matrix, alpha, colorSpace,connectivity, minAreaP, ac_morphology, SE1size, SE2size)
                        print("Computed gaussian modelling dataset num: "+str(dataset[cI])+" color space: "+colorSpace+" with alpha: "+str(alpha))

                    AUC[dataset[cI], areaI,SE1I,SE2I]=metrics.auc(R[dataset[cI], :], P[dataset[cI], :])
                    Pres[dataset[cI], areaI,SE1I,SE2I,:]=P[dataset[cI], :]
                    Recall[dataset[cI], areaI,SE1I,SE2I,:]=R[dataset[cI], :]
                    F1_score[dataset[cI], areaI,SE1I,SE2I,:]=F1[dataset[cI], :]
                    print("Gaussian modelling dataset num: "+str(dataset[cI])+" alpha: "+str(alpha)+" min_area: "+str(minAreaP)+" SE1: "+str(SE1size)+" SE2: "+str(SE2size)+"... done. AUC: "+str(AUC[dataset[cI], areaI,SE1I,SE2I])+"\n")

    for cI, colorSpace in enumerate(colorSpaces):
        alpha = alphas[cI]
        for areaI in range(len(minAreaPixels)):
            for SE1I in range(len(SE1sizes)):
                for SE2I in range(len(SE2sizes)):
                    plt.plot(alphas,F1_score[dataset[cI], areaI,SE1I,SE2I,:],label='F1-Dataset'+str(dataset[cI])+'_'+colorSpace+'_Area'+str(minAreaPixels[areaI])+'_SE1'+str(SE1sizes[SE1I])+'_SE2'+str(SE1sizes[SE2I]))
            plt.xlabel('alpha')
            plt.legend()
            plt.savefig('F1-Dataset'+str(dataset[cI])+'_'+colorSpace+'_Area'+str(minAreaPixels[areaI])+'.png')


    for cI, colorSpace in enumerate(colorSpaces):
        alpha = alphas[cI]
        for areaI in range(len(minAreaPixels)):
            for SE1I in range(len(SE1sizes)):
                for SE2I in range(len(SE2sizes)):
                    plt.plot(Recall[dataset[cI], areaI,SE1I,SE2I,:],Pres[dataset[cI], areaI,SE1I,SE2I,:],label='Dataset'+str(dataset[cI])+'_'+colorSpace+'_Area'+str(minAreaPixels[areaI])+'_SE1'+str(SE1sizes[SE1I])+'_SE2'+str(SE1sizes[SE2I])+' - AUC='+str(metrics.auc(Recall[dataset[cI], areaI,SE1I,SE2I,:],Pres[dataset[cI], areaI,SE1I,SE2I,:])))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.savefig('Dataset'+str(dataset[cI])+'_'+colorSpace+'_Area'+str(minAreaPixels[areaI])+'.png')
