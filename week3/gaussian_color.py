#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import math
import cv2
import numpy as np
from scipy import ndimage
from evaluate import *
from sklearn.metrics import confusion_matrix
from skimage.segmentation import clear_border
from PIL import Image
from skimage.measure import label
from skimage.measure import regionprops
from util import preprocess_pred_gt
from morphology import dilation, remove_dots, erosion
from hsv_shadow_remove import hsv_shadow_remove


# Define colors spaces to transform frames
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
shadow_removal = 1


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


def gaussian_color(path_test, path_gt, first_frame, last_frame, mu_matrix, sigma_matrix, alpha, colorSpace, connectivity, areaPixels,ac_morphology,SE1size,SE2size):

    """
    Description: gaussian 
    Input: path_test, path_gt, first_frame, last_frame, mu_matrix, sigma_matrix, alpha, colorSpace, connectivity, areaPixels
    Output: AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1
    """

    # Initialize metrics accumulators
    AccFP = 0
    AccFN = 0
    AccTP = 0
    AccTN = 0
    AccP = []
    AccR = []
    AccF1 = []

    # Initialize index to accumulate images
    index = 0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path+"gaussian_color_"+str(path_test.split("/")[1])+"_connectivity_"+str(connectivity)+".avi", fourcc, 60, (get_accumulator(path_test).shape[1], get_accumulator(path_test).shape[0]))
    out_noshadow = cv2.VideoWriter(video_path + "mask" +str(path_test.split("/")[1])+"_connectivity_"+str(connectivity)+".avi", fourcc, 60, (get_accumulator(path_test).shape[1], get_accumulator(path_test).shape[0]))

    # Define structuring element according to connectivity
    structuring_element = [[0,0,0],[0,0,0],[0,0,0]]
    if connectivity == '4':
        structuring_element = [[0,1,0],[1,1,1],[0,1,0]]
    if connectivity == '8':
        structuring_element = [[1,1,1],[1,1,1],[1,1,1]]

    # Read sequence of images sorted
    for filename in sorted(os.listdir(path_test)):
       
        # Check that frame is into range
        frame_num = int(filename[2:8])
        if frame_num >= first_frame and frame_num <= last_frame:

            # Read image from groundtruth 
            frame = cv2.imread(path_test+filename)
            # Check and transform color space 
            if colorSpace != 'RGB':
                frame = cv2.cvtColor(frame, colorSpaceConversion[colorSpace])
            # Compute pixels that belongs to background
            background = np.prod(abs(frame - mu_matrix) >= alpha*(sigma_matrix+2),axis=2)
            background_mask = background ##ADDED AUX VARIABLE
            # Convert bool to int values
            background = background.astype(int)
            # Replace 1 by 255
            background[background == 1] = 255
            # Scales, calculates absolute values, and converts the result to 8-bit
            background = cv2.convertScaleAbs(background)

            # Read groundtruth image
            gt = cv2.imread(path_gt+"gt"+filename[2:8]+".png", 0)

            # Shadow removal
            if shadow_removal == 1:
                shadow_mask = hsv_shadow_remove(cv2.imread(path_test + filename), mu_matrix)
                # Convert Boolean to 0, 1
                shadow_mask = 1*shadow_mask

                not_mask = np.logical_not(shadow_mask)

                background_noshadow = np.logical_and(not_mask, background_mask)
                background_noshadow = background_noshadow.astype(int)
                # Replace 1 by 255
                background_noshadow[background_noshadow == 1] = 255
                # Scales, calculates absolute values, and converts the result to 8-bit
                background_noshadow = cv2.convertScaleAbs(background_noshadow)

                background_frame_noshadow = cv2.cvtColor(background_noshadow, cv2.COLOR_GRAY2RGB)


                shadow_mask = shadow_mask.astype(int)
                shadow_mask[shadow_mask == 1] = 255
                shadow_mask = cv2.convertScaleAbs(shadow_mask)
                shadow_mask = cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2RGB)
                out_noshadow.write(shadow_mask)


                #out_noshadow.write(background_frame_noshadow)
                background = background_noshadow

            # Hole filling
            background = ndimage.binary_fill_holes(background, structure=structuring_element).astype(int)
            if ac_morphology==1:
                background = dilation(background,SE1size)
                background = ndimage.binary_fill_holes(background, structure=structuring_element).astype(int)
                background = erosion(background,SE1size)
                background = remove_dots(background,SE2size)

            # Replace 1 by 255
            background[background == 1] = 255
            # Scales, calculates absolute values, and converts the result to 8-bit
            background = cv2.convertScaleAbs(background)

            # Area filltering, label background regions
            label_image = label(background)
            # Measure properties of labeled background regions
            if areaPixels > 0:
                for region in regionprops(label_image):
                    # Remove regions smaller than fixed area
                    if region.area < areaPixels:
                        minr, minc,  maxr, maxc = region.bbox
                        background[minr:maxr,minc:maxc] = 0

            bck, gt = preprocess_pred_gt(background, gt)
            # Evaluate results
            TP, FP, TN, FN = evaluate_sample(bck, gt)

            # Accumulate metrics
            AccTP = AccTP + TP
            AccTN = AccTN + TN
            AccFP = AccFP + FP
            AccFN = AccFN + FN

            # Write frame into video
            video_frame = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
            out.write(video_frame)

    # Compute metrics
    print(" AccTP: {}  AccFP: {}  AccFN: {}".format(AccTP, AccFP, AccFN))
    if AccTP+AccFP == 0:
        AccP = 0
    else:
        AccP = AccTP / float(AccTP + AccFP)
    if AccTP + AccFN == 0:
        AccR = 0
    else:
        AccR = AccTP / float(AccTP + AccFN)
    if AccR == 0 and AccP == 0:
        AccF1 = 0
    else:
        AccF1 = 2 * AccP * AccR / (AccP + AccR)

    return AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1


