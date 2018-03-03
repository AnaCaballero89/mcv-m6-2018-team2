#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis. Task 2"

# Import libraries
import os
import math
import cv2
import numpy as np

# Path to save images and videos
images_path = "std-mean-images/"
video_path = "background-subtraction-videos/"

# Highway sequences configuration, range 1050 - 1350
highway_path = "datasets/highway/input/"	
highway_alpha = 2.5
highway_rho = 0.2

# Fall sequences configuration, range 1460 - 1560
fall_path = "datasets/fall/input/"  
fall_alpha = 2.5
fall_rho = 0.5

# Traffic sequences configuration, range 950 - 1050
traffic_path = "datasets/traffic/input/"
traffic_alpha = 3.25
traffic_rho = 0.15


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
    mean_matrix = cv2.convertScaleAbs(mean_matrix)
    cv2.imwrite(images_path+str(path_test.split("/")[1])+"_training_mean.png", mean_matrix)

    # Compute standard deviation matrix using numpy function
    std_matrix = np.std(accumulator, axis=2)
    std_matrix = cv2.convertScaleAbs(std_matrix)
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
  

def adaptive(path_test, first_frame, last_frame, mu_matrix, sigma_matrix, alpha, rho):

    """
    Description: background adapts
    Input: path_test, first_frame, last_frame,  mean_matrix, std_matrix, alpha, rho
    Output: None
    """

    # Initialize index to accumulate images
    index = 0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path+"adaptive_"+str(path_test.split("/")[1])+".avi", fourcc, 60, (get_accumulator(path_test).shape[1], get_accumulator(path_test).shape[0]))

    # Read sequence of images sorted
    for filename in sorted(os.listdir(path_test)):
       
        # Check that frame is into range
        frame_num = int(filename[2:8])
        if frame_num >= first_frame and frame_num <= last_frame:

            # Read image from groundtruth in grayscale
            frame = cv2.imread(path_test+filename, 0)

            # Check pixels that belongs to background
            background = abs(frame - mu_matrix) >= alpha*(sigma_matrix+2);
            # Convert bool to int values
            background = background.astype(int)
            # Replace 1 by 255
            background[background == 1] = 255
            # Scales, calculates absolute values, and converts the result to 8-bit
            background = cv2.convertScaleAbs(background)
            # Get foreground pixels
            foreground = cv2.bitwise_not(background)

            # Write frame into video
            video_frame = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
            out.write(video_frame)	

            # Apply background mask to frame image to retrieve only background grayscale pixels 
            background = cv2.bitwise_and(frame, frame, mask = background)
            # Apply foreground mask to frame image to retrieve only foreground grayscale pixels 
            foreground = cv2.bitwise_and(frame, frame, mask = foreground)

            # Compute mu matrix on all background pixels
            mu_matrix = ((rho*background)+((1-rho)*mu_matrix));
            # Add foreground pixels
            mu_matrix = mu_matrix + foreground
            # Scales, calculates absolute values, and converts the result to 8-bit
            mu_matrix = cv2.convertScaleAbs(mu_matrix)

            # Compute sigma matrix on all background pixels
            sigma_matrix = (rho**pow((background-mu_matrix),2))+((1-rho)**pow(sigma_matrix,2))
            # Add foreground pixels
            sigma_matrix = sigma_matrix + foreground
            # Scales, calculates absolute values, and converts the result to 8-bit
            sigma_matrix = cv2.convertScaleAbs(sigma_matrix)


if __name__ == "__main__":

    # Adaptive modelling
    # First 50% frames for training
    # Second 50% left backgrounds adapts

    mu_matrix, sigma_matrix = training(highway_path, 1050, 1199, highway_alpha);
    adaptive(highway_path, 1200, 1349, mu_matrix, sigma_matrix, highway_alpha, highway_rho); 

    mu_matrix, sigma_matrix = training(fall_path, 1460, 1509, fall_alpha);
    adaptive(fall_path, 1510, 1559, mu_matrix, sigma_matrix, fall_alpha, fall_rho);   

    mu_matrix, sigma_matrix = training(traffic_path, 950, 999, traffic_alpha);
    adaptive(traffic_path, 1000, 1049, mu_matrix, sigma_matrix, traffic_alpha, traffic_rho);
   

