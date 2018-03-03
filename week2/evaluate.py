#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import cv2
import os
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

    # True Positive (TP)
    TP = np.sum(np.logical_and(mask == 1, gt == 1))
    # True Negative (TN)
    TN = np.sum(np.logical_and(mask == 0, gt == 0))
    # False Positive (FP)
    FP = np.sum(np.logical_and(mask == 1, gt == 0))
    # False Negative (FN)
    FN = np.sum(np.logical_and(mask == 0, gt == 1))

    # Precision (P) 
    if float(np.count_nonzero(mask)) != 0.0:
            P = TP / float(TP + FP)
    else:  
        P = 0
    # Recall (R)
    if float(np.count_nonzero(gt)) != 0.0:
         R = TP / float(TP + FN)
    else: 
        R = 0
    # F1 score (F1)
    if float(P + R) != 0.0:
        F1 = 2 * P * R / (P + R)
    else: 
        F1 = 0

    return TP, FP, TN, FN, P, R, F1


