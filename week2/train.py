#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis. Task 1"

# Import libraries
import os
import math
import cv2
import numpy as np

# Path to save images and videos
images_path = "std-mean-images/"
video_path = "background-subtraction-videos/"


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


def training(path_test, first_frame, last_frame, alpha):

    """
    Description: train background subtraction
    Input: path_test, first_frame, last_frame, alpha
    Output: mean and standard deviation matrices
    """

    # Get accumulator of images
    accumulator = get_accumulator(path_test)

    # Initialize index to accumulate images
    index = 0

    # Read sequence of images sorted
    for filename in sorted(os.listdir(path_test)):
 
        # Check that frame is into range
        frame_num = int(filename[2:8])
        if frame_num >= first_frame and frame_num <= last_frame:

            # Read image from groundtruth in grayscale
            frame = cv2.imread(path_test+filename, 0)

            # Accumulate image into vector
            accumulator[..., index] = frame
            index = index + 1

    # Compute mean matrix using numpy function
    mean_matrix = np.mean(accumulator, axis=2)
    cv2.imwrite(images_path+str(path_test.split("/")[1])+"_training_mean.png", mean_matrix)

    # Compute standard deviation matrix using numpy function
    std_matrix = np.std(accumulator, axis=2)
    cv2.imwrite(images_path+str(path_test.split("/")[1])+"_training_std.png", std_matrix)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path+"non-adaptive_"+str(path_test.split("/")[1])+".avi", fourcc, 60, (accumulator.shape[1], accumulator.shape[0]))

    # Read sequence of images sorted to write video
    for filename in sorted(os.listdir(path_test)):
 
        # Check that frame is into range
        frame_num = int(filename[2:8])
        if frame_num >= first_frame and frame_num <= last_frame:

            # Read image from groundtruth in grayscale
            frame = cv2.imread(path_test+filename, 0)

            # Check pixels that belongs to background
            background = abs(frame - mean_matrix) >= alpha*(std_matrix+2);
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
   

