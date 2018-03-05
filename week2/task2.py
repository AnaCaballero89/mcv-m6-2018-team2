#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis. Task 2"

# Import libraries
import os
from train import *
from adaptive import *

# Highway sequences configuration, range 1050 - 1350
highway_path_in = "datasets/highway/input/"	
highway_path_gt = "datasets/highway/groundtruth/"
highway_alpha = 2.5
highway_rho = 0.2

# Fall sequences configuration, range 1460 - 1560
fall_path_in = "datasets/fall/input/"  
fall_path_gt = "datasets/fall/groundtruth/"  
fall_alpha = 2.5
fall_rho = 0.5

# Traffic sequences configuration, range 950 - 1050
traffic_path_in = "datasets/traffic/input/"
traffic_path_gt = "datasets/traffic/groundtruth/"
traffic_alpha = 3.25
traffic_rho = 0.15


if __name__ == "__main__":

    # Adaptive modelling
    # First 50% frames for training
    # Second 50% left backgrounds adapts

    print "Computing adaptive modelling on highway dataset..."
    mu_matrix, sigma_matrix = training(highway_path_in, 1050, 1199, highway_alpha);
    adaptive(highway_path_in, 1200, 1349, mu_matrix, sigma_matrix, highway_alpha, highway_rho); 
    print "Computing adaptive modelling on hihighway dataset... done"

    print "Computing adaptive modelling on fall dataset..."
    mu_matrix, sigma_matrix = training(fall_path_in, 1460, 1509, fall_alpha);
    adaptive(fall_path_in, 1510, 1559, mu_matrix, sigma_matrix, fall_alpha, fall_rho);   
    print "Computing adaptive modelling on fall dataset... done"

    print "Computing adaptive modelling on traffic dataset..."
    mu_matrix, sigma_matrix = training(traffic_path_in, 950, 999, traffic_alpha);
    adaptive(traffic_path_in, 1000, 1049, mu_matrix, sigma_matrix, traffic_alpha, traffic_rho);
    print "Computing adaptive modelling on traffic dataset... done"

