#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from util import *
from evaluate import evaluate_sample
import cv2
from sklearn import metrics


# Highway sequences configuration, range 1050 - 1350
highway_path_in = "./highway/input/"
highway_path_gt = "./highway/groundtruth/"
highway_in_files = os.listdir(highway_path_in)
highway_gt_files = os.listdir(highway_path_gt)
# Fall sequences configuration, range 1460 - 1560
fall_path_in = "./fall/input/"
fall_path_gt = "./fall/groundtruth/"
fall_in_files = os.listdir(fall_path_in)
fall_gt_files = os.listdir(fall_path_gt)
# Traffic sequences configuration, range 950 - 1050
traffic_path_in = "./traffic/input/"
traffic_path_gt = "./traffic/groundtruth/"
traffic_in_files = os.listdir(traffic_path_in)
traffic_gt_files = os.listdir(traffic_path_in)
video_path = "./background-subtraction-videos/"

# Define groundtruth labels namely
STATIC = 0
HARD_SHADOW = 50
OUTSIDE_REGION = 85
UNKNOW_MOTION = 170
MOTION = 255


def get_accumulator(path_test):

    """
    Description: get accumulator structure data
    Depends on image size to define borders
    Data are coded into 32 bits of floats
    Input: path test
    Output: accumulator
    """

    # Initialize accumualtor
    accumulator = np.zeros((0,0), np.float32)

    # Set accumulator depending on dataset choosen
    if path_test == "./highway/input/":
        accumulator = np.zeros((240,320,150), np.float32)
    if path_test == "./fall/input/":
        accumulator = np.zeros((480,720,50), np.float32)
    if path_test == "./traffic/input/":
        accumulator = np.zeros((240,320,50), np.float32)

    return accumulator


# Define thresholds
fgbg = cv2.createBackgroundSubtractorMOG()
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path + "mog" + str(highway_path_in.split("/")[1]) + ".avi", fourcc, 60,
                      (get_accumulator(highway_path_in).shape[1], get_accumulator(highway_path_in).shape[0]))
# Initialize metrics accumulators
AccFP = 0
AccFN = 0
AccTP = 0
AccTN = 0
#highway
print("Starting computing Background Subtractor MOG for Highway Dataset...")
for inp, gt in zip(sorted(highway_in_files), sorted(highway_gt_files)):
    # Check that frame is into range
    frame_num = int(inp[2:8])
    if frame_num >= 1200 and frame_num <= 1349:
        path_in = os.path.join(highway_path_in, inp)
        path_gt = os.path.join(highway_path_gt, gt)
        frame = cv2.imread(path_in, 0)
        groundtruth = cv2.imread(path_gt, 0)
        fgmask = fgbg.apply(frame)

        background = fgmask.flatten()
        gt = groundtruth.flatten()
        index2remove = [index for index, gt in enumerate(gt)
                        if gt == UNKNOW_MOTION or gt == OUTSIDE_REGION]
        gt = np.delete(gt, index2remove)
        gt[gt == HARD_SHADOW] = 0
        background = np.delete(background, index2remove)

        # Evaluate results
        TP, FP, TN, FN = evaluate_sample(background, gt)

        # Accumulate metrics
        AccTP = AccTP + TP
        AccTN = AccTN + TN
        AccFP = AccFP + FP
        AccFN = AccFN + FN
        # Write frame into video
        video_frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
        out.write(video_frame)

P, R, F1, FPR = get_metrics(AccTP, AccTN, AccFP, AccFN)
AUC1 = metrics.auc(FPR, R)
print("Background Subtractor in Highway --> Precision: {} , Recall: {} , F-Score: {}, AUC:  {}".
      format(P, R, F1, AUC1))


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path + "mog_" + str(fall_path_in.split("/")[1]) + ".avi", fourcc, 60,
                      (get_accumulator(fall_path_in).shape[1], get_accumulator(fall_path_in).shape[0]))
# Initialize metrics accumulators
AccFP = 0
AccFN = 0
AccTP = 0
AccTN = 0
#fall
print("Starting computing Background Subtractor MOG for Fall Dataset...")
for inp, gt in zip(sorted(fall_in_files), sorted(fall_gt_files)):
    # Check that frame is into range
    frame_num = int(inp[2:8])
    if frame_num >= 1510 and frame_num <= 1559:
        path_in = os.path.join(fall_path_in, inp)
        path_gt = os.path.join(fall_path_gt, gt)
        frame = cv2.imread(path_in, 0)
        groundtruth = cv2.imread(path_gt, 0)
        fgmask = fgbg.apply(frame)

        background = fgmask.flatten()
        gt = groundtruth.flatten()
        index2remove = [index for index, gt in enumerate(gt)
                        if gt == UNKNOW_MOTION or gt == OUTSIDE_REGION]
        gt = np.delete(gt, index2remove)
        gt[gt == HARD_SHADOW] = 0
        background = np.delete(background, index2remove)

        # Evaluate results
        TP, FP, TN, FN = evaluate_sample(background, gt)

        # Accumulate metrics
        AccTP = AccTP + TP
        AccTN = AccTN + TN
        AccFP = AccFP + FP
        AccFN = AccFN + FN
        # Write frame into video
        video_frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
        out.write(video_frame)

P, R, F1, FPR = get_metrics(AccTP, AccTN, AccFP, AccFN)
AUC1 = metrics.auc(FPR, R)
print("Background Subtractor in Fall --> Precision: {} , Recall: {} , F-Score: {}, AUC:  {}".
      format(P, R, F1, AUC1))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path + "mog_" + str(traffic_path_in.split("/")[1]) + ".avi", fourcc, 60,
                      (get_accumulator(traffic_path_in).shape[1], get_accumulator(traffic_path_in).shape[0]))
# Initialize metrics accumulators
AccFP = 0
AccFN = 0
AccTP = 0
AccTN = 0
# fall
print("Starting computing Background Subtractor MOG for Traffic Dataset...")
for inp, gt in zip(sorted(traffic_in_files), sorted(traffic_gt_files)):
    # Check that frame is into range
    frame_num = int(inp[2:8])
    if frame_num >= 1000 and frame_num <= 1049:
        path_in = os.path.join(traffic_path_in, inp)
        path_gt = os.path.join(traffic_path_gt, gt)
        frame = cv2.imread(path_in, 0)
        groundtruth = cv2.imread(path_gt, 0)
        fgmask = fgbg.apply(frame)

        background = fgmask.flatten()
        gt = groundtruth.flatten()
        index2remove = [index for index, gt in enumerate(gt)
                        if gt == UNKNOW_MOTION or gt == OUTSIDE_REGION]
        gt = np.delete(gt, index2remove)
        gt[gt == HARD_SHADOW] = 0
        background = np.delete(background, index2remove)

        # Evaluate results
        TP, FP, TN, FN = evaluate_sample(background, gt)

        # Accumulate metrics
        AccTP = AccTP + TP
        AccTN = AccTN + TN
        AccFP = AccFP + FP
        AccFN = AccFN + FN
        # Write frame into video
        video_frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
        out.write(video_frame)

P, R, F1, FPR = get_metrics(AccTP, AccTN, AccFP, AccFN)
AUC1 = metrics.auc(FPR, R)
print("Background Subtractor in Traffic --> Precision: {} , Recall: {} , F-Score: {}, AUC:  {}".
      format(P, R, F1, AUC1))