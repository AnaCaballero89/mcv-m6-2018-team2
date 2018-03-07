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
traffic_gt_files = os.listdir(traffic_path_gt)
video_path = "./background-subtraction-videos/"

# Define groundtruth labels namely
STATIC = 0
HARD_SHADOW = 50
OUTSIDE_REGION = 85
UNKNOW_MOTION = 170
MOTION = 255
vec_FP = []
vec_FN = []
vec_TP = []
vec_TN = []
vec_P1 = []
vec_P2 = []
vec_P3 = []
vec_R1 = []
vec_R2 = []
vec_R3 = []
vec_F11 = []
vec_F12 = []
vec_F13 = []
vec_FPR1 = []
vec_FPR2 = []
vec_FPR3 = []

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
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(video_path + "mog" + str(highway_path_in.split("/")[1]) + ".avi", fourcc, 60,
#                      (get_accumulator(highway_path_in).shape[1], get_accumulator(highway_path_in).shape[0]))
# Initialize metrics accumulators
AccFP = 0
AccFN = 0
AccTP = 0
AccTN = 0
alphas = np.arange(0, 1.1, 0.1)
fgbg = cv2.BackgroundSubtractorMOG()
#highway
print("Starting computing Background Subtractor MOG for Highway Dataset...")
vec_FP, vec_FN, vec_TP, vec_TN = init_vectors()

for alpha in alphas:
    print("ALPHA: {}".format(alpha))
    for inp, gt in zip(sorted(highway_in_files), sorted(highway_gt_files)):
        # Check that frame is into range
        frame_num = int(inp[2:8])
        if frame_num >= 1200 and frame_num <= 1349:
            path_in = os.path.join(highway_path_in, inp)
            path_gt = os.path.join(highway_path_gt, gt)
            frame = cv2.imread(path_in, 0)
            groundtruth = cv2.imread(path_gt, 0)
            fgmask = fgbg.apply(frame, learningRate=alpha)
            fgmask = fgmask.astype(int)
            background = fgmask.flatten()
            groundtruth = groundtruth.flatten()
            index2remove = [index for index, g in enumerate(groundtruth)
                            if g == UNKNOW_MOTION or g == OUTSIDE_REGION ]
            groundtruth = [0 if x == HARD_SHADOW else x for idx, x in enumerate(groundtruth)]
            groundtruth = np.delete(groundtruth, index2remove)
            background = np.delete(background, index2remove)

            # Evaluate results
            TP, FP, TN, FN = evaluate_sample(background, groundtruth)
            # Accumulate metrics
            AccTP = AccTP + TP
            AccTN = AccTN + TN
            AccFP = AccFP + FP
            AccFN = AccFN + FN
            # Write frame into video
            #video_frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
            #out.write(video_frame)
    vec_FP, vec_FN, vec_TP, vec_TN, vec_P1, vec_R1, vec_F11, vec_FPR1 = accumulate_values(vec_FP, vec_FN, vec_TP, vec_TN, vec_P1, vec_R1, vec_F11, vec_FPR1, AccFP, AccFN, AccTP, AccTN)
print 'Finishe computing on highway'

# Define the codec and create VideoWriter object
"""fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path + "mog_" + str(fall_path_in.split("/")[1]) + ".avi", fourcc, 60,
                      (get_accumulator(fall_path_in).shape[1], get_accumulator(fall_path_in).shape[0]))"""
# Initialize metrics accumulators
AccFP = 0
AccFN = 0
AccTP = 0
AccTN = 0
#fall
print("Starting computing Background Subtractor MOG for Fall Dataset...")
vec_FP, vec_FN, vec_TP, vec_TN = init_vectors()

for alpha in alphas:
    print("ALPHA: {}".format(alpha))
    for inp, gt in zip(sorted(fall_in_files), sorted(fall_gt_files)):
        # Check that frame is into range
        frame_num = int(inp[2:8])
        if frame_num >= 1510 and frame_num <= 1559:
            path_in = os.path.join(fall_path_in, inp)
            path_gt = os.path.join(fall_path_gt, gt)
            frame = cv2.imread(path_in, 0)
            groundtruth = cv2.imread(path_gt, 0)

            fgmask = fgbg.apply(frame, learningRate=alpha)
            fgmask = fgmask.astype(int)
            background = fgmask.flatten()
            groundtruth = groundtruth.flatten()
            index2remove = [index for index, g in enumerate(groundtruth)
                            if g == UNKNOW_MOTION or g == OUTSIDE_REGION ]
            groundtruth = [0 if x == HARD_SHADOW else x for idx, x in enumerate(groundtruth)]
            groundtruth = np.delete(groundtruth, index2remove)
            background = np.delete(background, index2remove)
            # Evaluate results
            TP, FP, TN, FN = evaluate_sample(background, groundtruth)

            # Accumulate metrics
            AccTP = AccTP + TP
            AccTN = AccTN + TN
            AccFP = AccFP + FP
            AccFN = AccFN + FN
            # Write frame into video
            #video_frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
            #out.write(video_frame)
    vec_FP, vec_FN, vec_TP, vec_TN, vec_P2, vec_R2, vec_F12, vec_FPR2 = accumulate_values(vec_FP, vec_FN, vec_TP, vec_TN, vec_P2, vec_R2, vec_F12, vec_FPR2, AccFP, AccFN, AccTP, AccTN)

print("Finished computing fall")
"""
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path + "mog_" + str(traffic_path_in.split("/")[1]) + ".avi", fourcc, 60,
                      (get_accumulator(traffic_path_in).shape[1], get_accumulator(traffic_path_in).shape[0]))
"""
# Initialize metrics accumulators
AccFP = 0
AccFN = 0
AccTP = 0
AccTN = 0
# fall
print("Starting computing Background Subtractor MOG for Traffic Dataset...")
vec_FP, vec_FN, vec_TP, vec_TN = init_vectors()

for alpha in alphas:
    print("ALPHA: {}".format(alpha))
    for inp, gt in zip(sorted(traffic_in_files), sorted(traffic_gt_files)):
        # Check that frame is into range
        frame_num = int(inp[2:8])
        if frame_num >= 1000 and frame_num <= 1049:
            path_in = os.path.join(traffic_path_in, inp)
            path_gt = os.path.join(traffic_path_gt, gt)
            frame = cv2.imread(path_in, 0)
            groundtruth = cv2.imread(path_gt, 0)

            fgmask = fgbg.apply(frame, learningRate=alpha)
            fgmask = fgmask.astype(int)
            background = fgmask.flatten()
            groundtruth = groundtruth.flatten()
            index2remove = [index for index, g in enumerate(groundtruth)
                            if g == UNKNOW_MOTION or g == OUTSIDE_REGION ]
            groundtruth = [0 if x == HARD_SHADOW else x for idx, x in enumerate(groundtruth)]
            groundtruth = np.delete(groundtruth, index2remove)
            background = np.delete(background, index2remove)
            # Evaluate results
            TP, FP, TN, FN = evaluate_sample(background, groundtruth)

            # Accumulate metrics
            AccTP = AccTP + TP
            AccTN = AccTN + TN
            AccFP = AccFP + FP
            AccFN = AccFN + FN
            # Write frame into video
            #video_frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
            #out.write(video_frame)

    vec_FP, vec_FN, vec_TP, vec_TN, vec_P3, vec_R3, vec_F13, vec_FPR3 = accumulate_values(vec_FP, vec_FN, vec_TP, vec_TN,
                                                                                      vec_P3, vec_R3, vec_F13, vec_FPR3,
                                                                                      AccFP, AccFN, AccTP, AccTN)
print("Finished computing on traffic")

# Plot recall on graph
plot_recall(vec_R1, vec_R2, vec_R3, alphas)

# Plot precision on graph
plot_precision(vec_P1, vec_P2, vec_P3, alphas)
# Plot fscore on graph
plot_fscore(vec_F11, vec_F12, vec_F13, alphas)
# plot precision-recall:
plot_PR_REC(vec_P1, vec_P2, vec_P3, vec_R1, vec_R2, vec_R3)
# PLOT ROC curves
plot_ROC(vec_R1, vec_FPR1, vec_R2, vec_FPR2, vec_R3, vec_FPR3)