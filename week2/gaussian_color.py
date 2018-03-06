#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import math
import cv2
import numpy as np
from evaluate_color import *
from sklearn.metrics import confusion_matrix

colorSpaceConversion={}
colorSpaceConversion['YCrCb'] = cv2.COLOR_BGR2YCR_CB
colorSpaceConversion['HSV']   = cv2.COLOR_BGR2HSV
colorSpaceConversion['gray'] = cv2.COLOR_BGR2GRAY

# Path to save images and videos
images_path = "std-mean-images/"
video_path = "background-subtraction-videos/"

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
    if path_test == "datasets/highway/input/":
        accumulator = np.zeros((240,320,150), np.float32)
    if path_test == "datasets/fall/input/":
        accumulator = np.zeros((480,720,50), np.float32)
    if path_test == "datasets/traffic/input/":
        accumulator = np.zeros((240,320,50), np.float32)

    return accumulator


def gaussian_color(path_test, path_gt, first_frame, last_frame, mu_matrix, sigma_matrix, alpha, colorSpace):

    """
    Description: gaussian 
    Input: path_test, path_gt, first_frame, last_frame, mu_matrix, sigma_matrix, alpha
    Output: AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1
    """

    # Initialize metrics accumulators
    AccFP = 0
    AccFN = 0
    AccTP = 0
    AccTN = 0
    AccP = 0
    AccR = 0
    AccF1 = 0

    # Initialize index to accumulate images
    index = 0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path+"gaussian_"+str(path_test.split("/")[1])+".avi", fourcc, 60, (get_accumulator(path_test).shape[1], get_accumulator(path_test).shape[0]))

    # Read sequence of images sorted
    for filename in sorted(os.listdir(path_test)):
       
        # Check that frame is into range
        frame_num = int(filename[2:8])
        if frame_num >= first_frame and frame_num <= last_frame:

            # Read image from groundtruth
            frame = cv2.imread(path_test+filename)
            if colorSpace != 'RGB':
                frame = cv2.cvtColor(frame, colorSpaceConversion[colorSpace])
            # Compute pixels that belongs to background
            background = np.prod(abs(frame - mu_matrix) >= alpha*(sigma_matrix+2),axis=2);
            # Convert bool to int values
            background = background.astype(int)
            # Replace 1 by 255
            background[background == 1] = 255
            # Scales, calculates absolute values, and converts the result to 8-bit
            background = cv2.convertScaleAbs(background)

            # Read groundtruth image
            gt = cv2.imread(path_gt+"gt"+filename[2:8]+".png",0)
            # Remember that we will use values as background 0 and 50, foreground 255, and unknow (not evaluated) 85 and 170
            # Replace values acording previous assumption
            gt[gt == HARD_SHADOW] = 0
            gt[gt == OUTSIDE_REGION] = 0
            gt[gt == UNKNOW_MOTION] = 0

            # Evaluate results
            TP, FP, TN, FN, Pim, Rim, F1im = evaluate_sample(background, gt)

            # Accumulate metrics
            AccTP = AccTP + TP
            AccTN = AccTN + TN
            AccFP = AccFP + FP
            AccFN = AccFN + FN

            # Write frame into video
            video_frame = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
            out.write(video_frame)	


    AccP = AccTP / float(AccTP + AccFP)
    AccR = AccTP / float(AccTP + AccFN)
    AccF1 = 2 * AccP * AccR / (AccP + AccR)

    return AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1


