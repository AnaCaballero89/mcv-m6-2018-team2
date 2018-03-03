#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis. Task 2"

# Import libraries
import os
import math
import cv2
import numpy as np

# Highway sequences configuration, range 1050 - 1350
highway_path = "datasets/highway/input/"	
highway_alhpa = 2.5
highway_rho = 0.5

# Fall sequences configuration, range 1460 - 1560
fall_path = "datasets/fall/input/"  
fall_alpha = 2.5
fall_rho = 0.5

# Traffic sequences configuration, range 950 - 1050
traffic_path = "datasets/traffic/input/"
traffic_alpha = 2.5
traffic_rho = 0.5


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


def training(path_test, first_frame, last_frame):

    """
    Description: train background subtraction
    Input: path test, first frame and last frame
    Output: mean and standard deviation matrices
    """

    # Get accumulator of images
    accumulator = get_accumulator(path_test)

    # Initialize index to accumulate images
    index = 0

    # Read sequence of images
    for filename in os.listdir(path_test):
       
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
    mean_matrix = cv2.convertScaleAbs(mean_matrix)
    cv2.imshow("mean matrix", mean_matrix)
    cv2.waitKey(0) 

    # Compute standard deviation matrix using numpy function
    std_matrix = np.std(accumulator, axis=2)
    std_matrix = cv2.convertScaleAbs(std_matrix)
    cv2.imshow("std matrix", std_matrix)
    cv2.waitKey(0)  

    return mean_matrix, std_matrix
  

def adapts(path_test, first_frame, last_frame,  mean_matrix, std_matrix, alpha, rho):

    """
    Description: background adapts
    Input: path_test, threshold, first_frame, last_frame,  mu, sigma, alpha, rho
    Output: None
    """

    # Read sequence of images
    for filename in os.listdir(path_test):
       
        # Check that frame is into range
        frame_num = int(filename[2:8])

        if frame_num >= first_frame and frame_num <= last_frame:

            # Read image from groundtruth in grayscale
            frame = cv2.imread(path_test+filename, 0)

            # TODO: Check pixels that belongs to background
            # TODO: Compute mu matrix 
            # TODO: Compute sigma matrix         
 

if __name__ == "__main__":

    # Adaptive modelling
    # First 50% frames for training
    # Second 50% left backgrounds adapts

    mean_matrix, std_matrix = training(highway_path, 1050, 1199);
    adapts(highway_path, 1199, 1350, mean_matrix, std_matrix, highway_alhpa, highway_rho);

    # mu, sigma = training(fall_path, 1460, 1509);
    # adapts(fall_path, 1509, 1560, mu, sigma, highway_alhpa, highway_rho);
    
    # mu, sigma = training(traffic_path, 950, 999);
    # adapts(traffic_path, 999, 1050, mu, sigma, highway_alhpa, highway_rho);


