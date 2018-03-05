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

def plot_f1_threshold(FP, FN, TP, TN, alphas, name):
    """
    Description: plot f1 vs threshold
    Input: FP, FN, TP, TN, alphas, name
    Output: None
    """
    fscore = []
    for i in range(len(alphas)):
        precision = TP[i] / float(TP[i] + FP[i])
        recall = TP[i] / float(TP[i] + FN[i])
        fscore.append(2 * precision * recall / (precision + recall))

    plt.title('F1 vs threshold metric '+name+' dataset')
    plt.xlabel('Threshold')
    plt.ylabel('F1')
    plt.ylim([0, 1])
    plt.xlim([0, max(alphas)])
    plt.plot(fscore)
    plt.show()




def plot_precision_recall_f1(FP, FN, TP, TN, alphas, name):
    fscore = []
    prec = []
    rec = []
    for i in range(len(alphas)):
        precision = TP[i] / float(TP[i] + FP[i])
        recall = TP[i] / float(TP[i] + FN[i])
        prec.append(precision)
        rec.append(recall)
        fscore.append(2 * precision * recall / (precision + recall))

    plt.title('Precision, Recall and F1 metric '+name+' dataset')
    plt.xlabel('Threshold')
    plt.ylabel('Metric')
    plt.ylim([0, 1])
    plt.xlim([0, max(alphas)])
    plt.plot(prec)
    plt.plot(rec)
    plt.plot(fscore)
    red_patch = mpatches.Patch(color='orange', label='Precision')
    blue_patch = mpatches.Patch(color='blue', label='Recall')
    green_patch = mpatches.Patch(color='green', label='F1')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.show()

def plot_precision_recall(FP, FN, TP, TN, alphas, name):
    prec = []
    rec = []
    for i in range(len(alphas)):
        precision = TP[i] / float(TP[i] + FP[i])
        recall = TP[i] / float(TP[i] + FN[i])
        prec.append(precision)
        rec.append(recall)
    print("Precision:")
    print(prec)
    print("Recall")
    print(rec)

    plt.title('Precision and Recall metric ' + name + ' dataset')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.plot(rec, prec)
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


