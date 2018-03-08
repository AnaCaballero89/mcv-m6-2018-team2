#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis. Task 1"

# Import libraries
import os
import math
import cv2
import numpy as np

# Define colors spaces to tranform frames
colorSpaceConversion={}
colorSpaceConversion['YCrCb'] = cv2.COLOR_BGR2YCR_CB
colorSpaceConversion['HSV']   = cv2.COLOR_BGR2HSV
colorSpaceConversion['gray'] = cv2.COLOR_BGR2GRAY

# Path to save images and videos
images_path = "std-mean-images/"
video_path = "background-subtraction-videos/"


def get_accumulator_color(path_test):

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


def training_color(path_test, first_frame, last_frame, alpha, colorSpace):

    """
    Description: train background subtraction
    Input: path_test, first_frame, last_frame, alpha
    Output: mean and standard deviation matrices
    """

    # Get accumulator of images by channels
    accumulator0 = get_accumulator_color(path_test)
    accumulator1 = get_accumulator_color(path_test)
    accumulator2 = get_accumulator_color(path_test)
    mean_matrix = np.zeros(accumulator0.shape[0:2]+tuple([3]), np.float32)
    std_matrix = np.zeros(accumulator0.shape[0:2]+tuple([3]), np.float32)

    # Initialize index to accumulate images
    index = 0

    # Read sequence of images sorted
    for filename in sorted(os.listdir(path_test)):
 
        # Check that frame is into range
        frame_num = int(filename[2:8])
        if frame_num >= first_frame and frame_num <= last_frame:
            # Read image from groundtruth in grayscale
            frame = cv2.imread(path_test+filename)
            # Check and transform color space 
            if colorSpace != 'RGB':
                frame = cv2.cvtColor(frame, colorSpaceConversion[colorSpace])
            # Accumulate image into vector
            accumulator0[..., index] = frame[...,0]
            accumulator1[..., index] = frame[...,1]
            accumulator2[..., index] = frame[...,2]
            index = index + 1

    # Compute mean matrix using numpy function among channels
    mean_matrix[:,:, 0] = np.mean(accumulator0, axis=2)
    mean_matrix[:,:, 1] = np.mean(accumulator1, axis=2)
    mean_matrix[:,:, 2] = np.mean(accumulator2, axis=2)
    cv2.imwrite(images_path+str(path_test.split("/")[1])+"_training_mean.png", mean_matrix)

    # Compute standard deviation matrix using numpy function among channels
    std_matrix[:,:, 0] = np.std(accumulator0, axis=2)
    std_matrix[:,:, 1] = np.std(accumulator1, axis=2)
    std_matrix[:,:, 2] = np.std(accumulator2, axis=2)
    cv2.imwrite(images_path+str(path_test.split("/")[1])+"_training_std.png", std_matrix)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path+"non-adaptive_"+str(path_test.split("/")[1])+".avi", fourcc, 60, (accumulator0.shape[1], accumulator0.shape[0]))

    # Read sequence of images sorted to write video
    for filename in sorted(os.listdir(path_test)):
 
        # Check that frame is into range
        frame_num = int(filename[2:8])
        if frame_num >= first_frame and frame_num <= last_frame:
            # Read image from groundtruth in grayscale
            frame = cv2.imread(path_test+filename)
            # Check and transform color space 
            if colorSpace != 'RGB':
                frame = cv2.cvtColor(frame, colorSpaceConversion[colorSpace])

            # Check pixels that belongs to background
            background = np.prod(abs(frame - mean_matrix) >= alpha*(std_matrix+2),axis=2)
            # Convert bool to int values
            background = background.astype(int)
            # Replace 1 by 255
            background[background == 1] = 255
            # Scales, calculates absolute values, and converts the result to 8-bit
            background = cv2.convertScaleAbs(background)

            # Write frame into video
            frame = cv2.cvtColor(background,cv2.COLOR_GRAY2RGB)
            out.write(frame)

    return mean_matrix, std_matrix
   

