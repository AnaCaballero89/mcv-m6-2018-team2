#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import cv2
import os
from sklearn.metrics import confusion_matrix
import numpy as np


def evaluate_sample(mask, gt):
 
    """
    Description: evaluate sample
    Input: mask (predicted labes) and gt (groundtruth labels)
    Output; TP, FP, TN, FN, P, R, F1
    """

    # Transform values 255 to 1
    mask = mask.clip(max=1)
    gt = gt.clip(max=1)

    # Compute confusion matrix to evaluate the accuracy of a classification   
    # Check that there more than one unique value to classify samples
    # Confusion matrix need more than 1 value to unpack function 
    # If there are one unique value, all values on sample correspond to background (0)
    if((len(np.unique(gt)) == 1) and (len(np.unique(mask)) == 1)):
        TN = 0
        FP = 0
        FN = 0
        TP = 0
    else:
        TN, FP, FN, TP = confusion_matrix(gt.flatten(), mask.flatten()).ravel()

    return TP, FP, TN, FN


