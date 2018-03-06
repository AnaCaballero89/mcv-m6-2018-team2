#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import plotly.tools as tls
from train import *
from gaussian import *
from util import *

# Highway sequences configuration, range 1050 - 1350
highway_path_in = "datasets/highway/input/"	
highway_path_gt = "datasets/highway/groundtruth/"

# Fall sequences configuration, range 1460 - 1560
fall_path_in = "datasets/fall/input/"  
fall_path_gt = "datasets/fall/groundtruth/"  

# Traffic sequences configuration, range 950 - 1050
traffic_path_in = "datasets/traffic/input/"
traffic_path_gt = "datasets/traffic/groundtruth/"

# Define thresholds
alphas = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

# Define accumulators
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

if __name__ == "__main__":

    # Gaussian modelling
    # First 50% of the test sequence to model background
    # Second 50% to segment the foreground

    print "Starting gaussian modelling on highway dataset computation..."
    for alpha in alphas:
        mean_matrix, std_matrix = training(highway_path_in, 1050, 1199, alpha);
        AccFP, AccFN, AccTP, AccTN= gaussian(highway_path_in, highway_path_gt, 1200, 1349, mean_matrix, std_matrix, alpha)
        vec_FP, vec_FN, vec_TP, vec_TN, vec_P1, vec_R1, vec_F11 = accumulate_values(vec_FP, vec_FN, vec_TP, vec_TN, vec_P1, vec_R1, vec_F11, AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1)
        print "Computed gaussian modelling on highway dataset with alpha "+str(alpha)
    print "Starting gaussian modelling on highway dataset computation... done\n"
    
    # Plot metrics on graph
    plot_graph_FP_FN_TP_TN(vec_FP, vec_FN, vec_TP, vec_TN, alphas, 'highway')

    # Initialize vectors
    vec_FP, vec_FN, vec_TP, vec_TN = init_vectors()

    print "Starting gaussian modelling on fall dataset computation..."
    for alpha in alphas:
        mean_matrix, std_matrix = training(fall_path_in, 1460, 1509, alpha);
        AccFP, AccFN, AccTP, AccTN = gaussian(fall_path_in, fall_path_gt, 1510, 1559, mean_matrix, std_matrix, alpha)
        vec_FP, vec_FN, vec_TP, vec_TN, vec_P2, vec_R2, vec_F12 = accumulate_values(vec_FP, vec_FN, vec_TP, vec_TN, vec_P2, vec_R2, vec_F12, AccFP, AccFN, AccTP, AccTN)
        print "Computed gaussian modelling on fall dataset with alpha "+str(alpha)
    print "Starting gaussian modelling on fall dataset computation... done\n"

    # Plot metrics on graph
    plot_graph_FP_FN_TP_TN(vec_FP, vec_FN, vec_TP, vec_TN, alphas, 'fall')

    # Initialize vectors
    vec_FP, vec_FN, vec_TP, vec_TN = init_vectors()

    print "Starting gaussian modelling on traffic dataset computation..."
    for alpha in alphas:
        mean_matrix, std_matrix = training(traffic_path_in, 950, 999, alpha);
        AccFP, AccFN, AccTP, AccTN = gaussian(traffic_path_in, traffic_path_gt, 1000, 1049, mean_matrix, std_matrix, alpha)
        vec_FP, vec_FN, vec_TP, vec_TN, vec_P3, vec_R3, vec_F13 = accumulate_values(vec_FP, vec_FN, vec_TP, vec_TN, vec_P3, vec_R3, vec_F13, AccFP, AccFN, AccTP, AccTN)
        print "Computed gaussian modelling on traffic with alpha "+str(alpha)
    print "Starting gaussian modelling on traffic dataset computation... done\n"

    # Plot metrics on graph
    plot_graph_FP_FN_TP_TN(vec_FP, vec_FN, vec_TP, vec_TN, alphas, 'traffic')

    # Plot recall on graph
    plot_recall(vec_R1, vec_R2, vec_R3, alphas)

    # Plot precision on graph
    plot_precision(vec_P1, vec_P2, vec_P3, alphas)


