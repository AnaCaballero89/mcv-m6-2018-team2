#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
from train import *
from gaussian import *

# Highway sequences configuration, range 1050 - 1350
highway_path_in = "datasets/highway/input/"	
highway_path_gt = "datasets/highway/groundtruth/"
highway_alpha = 2.5

# Fall sequences configuration, range 1460 - 1560
fall_path_in = "datasets/fall/input/"  
fall_path_gt = "datasets/fall/groundtruth/"  
fall_alpha = 2.5

# Traffic sequences configuration, range 950 - 1050
traffic_path_in = "datasets/traffic/input/"
traffic_path_gt = "datasets/traffic/groundtruth/"
traffic_alpha = 3.25


if __name__ == "__main__":

    # Gaussian modelling
    # First 50% of the test sequence to model background
    # Second 50% to segment the foreground

    print "Computing gaussian modelling on highway dataset..."
    mean_matrix, std_matrix = training(highway_path_in, 1050, 1199, highway_alpha);
    AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1 = gaussian(highway_path_in, highway_path_gt, 1200, 1349, mean_matrix, std_matrix, highway_alpha); 
    print "FP: "+str(AccFP)+", FN: "+str(AccFN)+", TP: "+str(AccTP)+", TN: "+str(AccTN)
    print "P: "+str(AccP)+", R: "+str(AccR)+", F1: "+str(AccF1)
    print "Computing gaussian modelling on highway dataset... done\n"

    print "Computing gaussian modelling on fall dataset..."
    mean_matrix, std_matrix = training(fall_path_in, 1460, 1509, fall_alpha);
    AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1 = gaussian(fall_path_in, fall_path_gt, 1510, 1559, mean_matrix, std_matrix, fall_alpha);  
    print "FP: "+str(AccFP)+", FN: "+str(AccFN)+", TP: "+str(AccTP)+", TN: "+str(AccTN)
    print "P: "+str(AccP)+", R: "+str(AccR)+", F1: "+str(AccF1)
    print "Computing gaussian modelling on fall dataset... done\n"

    print "Computing gaussian modelling on traffic dataset..."
    mean_matrix, std_matrix = training(traffic_path_in, 950, 999, traffic_alpha);
    AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1 = gaussian(traffic_path_in, traffic_path_gt, 1000, 1049, mean_matrix, std_matrix, traffic_alpha);
    print "FP: "+str(AccFP)+", FN: "+str(AccFN)+", TP: "+str(AccTP)+", TN: "+str(AccTN)
    print "P: "+str(AccP)+", R: "+str(AccR)+", F1: "+str(AccF1)
    print "Computing gaussian modelling on traffic dataset... done\n"


