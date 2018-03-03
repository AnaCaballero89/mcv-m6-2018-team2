#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_graph_FP_FN_TP_TN(FP, FN, TP, TN, alphas, name):

    """
    Description: plot graph
    Input: FP, FN, TP, TN
    Output: None
    """

    plt.title('Evaluate metrics on '+name+' dataset')
    plt.xlabel('Threshold') 
    plt.ylabel('Number of pixels')
    plt.ylim([0, max(FP)])
    plt.xlim([0, max(alphas)])
    plt.plot(FP, '-')
    plt.plot(FN, '-')
    plt.plot(TP, '-')
    plt.plot(TN, '-')
    red_patch = mpatches.Patch(color='red', label='FP')
    blue_patch = mpatches.Patch(color='blue', label='FN')
    green_patch = mpatches.Patch(color='green', label='TP')
    orange_patch = mpatches.Patch(color='orange', label='TN')
    plt.legend(handles=[red_patch, blue_patch, green_patch, orange_patch])
    plt.show()


def init_vectors():

    """
    Description: initialize vectors
    Input: None
    Output: None
    """

    vec_FP = []
    vec_FN = []
    vec_TP = []
    vec_TN = []
    vec_P = []
    vec_R = []
    vec_F1 = []

    return vec_FP, vec_FN, vec_TP, vec_TN, vec_P, vec_R, vec_F1


def accumulate_values(vec_FP, vec_FN, vec_TP, vec_TN, vec_P, vec_R, vec_F1, AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1):

    """
    Description: accumulate values
    Input: AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1
    Output: vec_FP, vec_FN, vec_TP, vec_TN, vec_P, vec_R, vec_F1
    """

    vec_FP.append(AccFP)
    vec_FN.append(AccFN)
    vec_TP.append(AccTP)
    vec_TN.append(AccTN)
    vec_P.append(AccP)
    vec_R.append(AccR)
    vec_F1.append(AccF1)  

    return vec_FP, vec_FN, vec_TP, vec_TN, vec_P, vec_R, vec_F1


