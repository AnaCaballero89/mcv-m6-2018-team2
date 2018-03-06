#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis. Task 2"

# Import libraries
import os
from train import *
from adaptive import *
from util import *

# Highway sequences configuration, range 1050 - 1350
highway_path_in = "./highway/input/"
highway_path_gt = "./highway/groundtruth/"
highway_alpha = 2.5
highway_rho = 0.2

# Fall sequences configuration, range 1460 - 1560
fall_path_in = "./fall/input/"
fall_path_gt = "./fall/groundtruth/"
fall_alpha = 2.5
fall_rho = 0.5

# Traffic sequences configuration, range 950 - 1050
traffic_path_in = "./traffic/input/"
traffic_path_gt = "./traffic/groundtruth/"
traffic_alpha = 3.25
traffic_rho = 0.15


if __name__ == "__main__":

    # Adaptive modelling
    # First 50% frames for training
    # Second 50% left backgrounds adapts

    print("Computing adaptive modelling on highway dataset...")
    mu_matrix, sigma_matrix = training(highway_path_in, 1050, 1199, highway_alpha)
    AccFP, AccFN, AccTP, AccTN = adaptive(highway_path_in, 1200, 1349, mu_matrix, sigma_matrix, highway_alpha, highway_rho)
    print("Computing adaptive modelling on hihighway dataset... done")

    print("Computing adaptive modelling on fall dataset...")
    mu_matrix, sigma_matrix = training(fall_path_in, 1460, 1509, fall_alpha)
    AccFP, AccFN, AccTP, AccTN = adaptive(fall_path_in, 1510, 1559, mu_matrix, sigma_matrix, fall_alpha, fall_rho)
    print("Computing adaptive modelling on fall dataset... done")

    print("Computing adaptive modelling on traffic dataset...")
    mu_matrix, sigma_matrix = training(traffic_path_in, 950, 999, traffic_alpha)
    AccFP, AccFN, AccTP, AccTN = adaptive(traffic_path_in, 1000, 1049, mu_matrix, sigma_matrix, traffic_alpha, traffic_rho)
    print("Computing adaptive modelling on traffic dataset... done")

    alphas = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    rhos = np.arange(0, 1.1, 0.1)
    vec_FP, vec_FN, vec_TP, vec_TN = init_vectors()
    best_F = 0
    best_rho = 0
    best_alpha = 0
    print("Computing grid search on highway dataset...")
    for rho in rhos:
        for alpha in alpha:
            mu_matrix, sigma_matrix = training(highway_path_in, 1050, 1199, alpha)
            AccFP, AccFN, AccTP, AccTN = adaptive(highway_path_in, 1200, 1349, mu_matrix, sigma_matrix,
                                                  alpha, rho)
            P, R, F1, FPR = get_metrics(AccTP, AccTN, AccFP, AccFN)
            if F1 > best_F:
                best_F = F1
                best_alpha = alpha
                best_rho = rho
    print('Highway: Best F-Score: {} , optimal alpha : {} , optimal_rho: {}').format(best_F, best_alpha, best_rho)
    print("Computing grid search on fall dataset...")
    for rho in rhos:
        for alpha in alpha:
            mu_matrix, sigma_matrix = training(fall_path_in, 1050, 1199, alpha)
            AccFP, AccFN, AccTP, AccTN = adaptive(fall_path_in, 1200, 1349, mu_matrix, sigma_matrix,
                                                  alpha, rho)
            P, R, F1, FPR = get_metrics(AccTP, AccTN, AccFP, AccFN)
            if F1 > best_F:
                best_F = F1
                best_alpha = alpha
                best_rho = rho
    print('Fall: Best F-Score: {} , optimal alpha : {} , optimal_rho: {}').format(best_F, best_alpha, best_rho)
    print("Computing grid search on traffic dataset...")
    for rho in rhos:
        for alpha in alpha:
            mu_matrix, sigma_matrix = training(traffic_path_in, 1050, 1199, alpha)
            AccFP, AccFN, AccTP, AccTN = adaptive(traffic_path_in, 1200, 1349, mu_matrix, sigma_matrix,
                                                  alpha, rho)
            P, R, F1, FPR = get_metrics(AccTP, AccTN, AccFP, AccFN)
            if F1 > best_F:
                best_F = F1
                best_alpha = alpha
                best_rho = rho
    print('Traffic: Best F-Score: {} , optimal alpha : {} , optimal_rho: {}').format(best_F, best_alpha, best_rho)
