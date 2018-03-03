#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis. Task 1"

# Import libraries
import os
import math
import cv2
import numpy as np
from train import *
from gaussian import *

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


if __name__ == "__main__":

    # Adaptive modelling
    # First 50% of the test sequence to model background
    # Second 50% to segment the foreground

    mean_matrix, std_matrix = training(highway_path, 1050, 1199, highway_alpha);
    gaussian(highway_path, 1200, 1349, mean_matrix, std_matrix, highway_alpha, highway_rho); 

    mean_matrix, std_matrix = training(fall_path, 1460, 1509, fall_alpha);
    gaussian(fall_path, 1510, 1559, mean_matrix, std_matrix, fall_alpha, fall_rho);   

    mean_matrix, std_matrix = training(traffic_path, 950, 999, traffic_alpha);
    gaussian(traffic_path, 1000, 1049, mean_matrix, std_matrix, traffic_alpha, traffic_rho);
   

