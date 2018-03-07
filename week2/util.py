#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics


def plot_fscore(vec_F1, vec_F2, vec_F3, alphas):

    """
    Description: plot recall
    Input: vec_R1, vec_R2, vec_R3, alphas
    Output: None
    """

    plt.clf()
    plt.title('Recall vs alpha')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.ylim([0, max(max(vec_F1),max(vec_F2),max(vec_F3))])
    plt.xlim([0, max(alphas)])
    plt.plot(vec_F1, 'r-')
    plt.plot(vec_F2, 'b-')
    plt.plot(vec_F3, 'g-')
    red_patch = mpatches.Patch(color='red', label='highway')
    blue_patch = mpatches.Patch(color='blue', label='fall')
    green_patch = mpatches.Patch(color='green', label='traffic')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.savefig()

def plot_recall(vec_R1, vec_R2, vec_R3, alphas):

    """
    Description: plot recall
    Input: vec_R1, vec_R2, vec_R3, alphas
    Output: None
    """

    plt.clf()
    plt.title('Recall vs alpha')
    plt.xlabel('Threshold') 
    plt.ylabel('Recall')
    plt.ylim([0, max(max(vec_R1),max(vec_R2),max(vec_R3))])
    plt.xlim([0, max(alphas)])
    plt.plot(vec_R1, 'r-')
    plt.plot(vec_R2, 'b-')
    plt.plot(vec_R3, 'g-')
    red_patch = mpatches.Patch(color='red', label='highway')
    blue_patch = mpatches.Patch(color='blue', label='fall')
    green_patch = mpatches.Patch(color='green', label='traffic')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    #plt.show()
    plt.savefig('recall.png')


def plot_precision(vec_P1, vec_P2, vec_P3, alphas):

    """
    Description: plot precision
    Input: vec_P1, vec_P2, vec_P3, alphas
    Output: None
    """

    plt.clf()
    plt.title('Precision vs alpha')
    plt.xlabel('Threshold') 
    plt.ylabel('Precision')
    plt.ylim([0, max(max(vec_P1),max(vec_P2),max(vec_P3))])
    plt.xlim([0, max(alphas)])
    plt.plot(vec_P1, 'r-')
    plt.plot(vec_P2, 'b-')
    plt.plot(vec_P3, 'g-')
    red_patch = mpatches.Patch(color='red', label='highway')
    blue_patch = mpatches.Patch(color='blue', label='fall')
    green_patch = mpatches.Patch(color='green', label='traffic')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    #plt.show()
    plt.savefig('precision.png')



def plot_graph_FP_FN_TP_TN(FP, FN, TP, TN, alphas, name):

    """
    Description: plot graph
    Input: FP, FN, TP, TN
    Output: None
    """

    plt.clf()
    plt.title('Evaluate metrics on '+name+' dataset')
    plt.xlabel('Threshold') 
    plt.ylabel('Number of pixels')
    plt.ylim([0, max(max(FP), max(FN), max(TP), max(TN))])
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
    #plt.show()
    plt.savefig('tp_fn_fp_tn'+name+'.png')

def plot_PR_REC(vec_P1, vec_P2, vec_P3, vec_R1, vec_R2, vec_R3):
    plt.clf()
    plt.title('Precision and recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.plot(vec_R1, vec_P1)
    plt.plot(vec_R2, vec_P2)
    plt.plot(vec_R3, vec_P3)
    red_patch = mpatches.Patch(color='red', label='highway')
    blue_patch = mpatches.Patch(color='blue', label='fall')
    green_patch = mpatches.Patch(color='green', label='traffic')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    #plt.show()
    plt.savefig('precision_recall_overall.png')

def plot_metrics_alpha(prec,rec,f1, alphas, name):

    """
    Description: plot graph
    Input: FP, FN, TP, TN
    Output: None
    """

    plt.clf()
    plt.title('Metrics on '+name+' dataset')
    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.ylim([0, 1])
    plt.xlim([0, max(alphas)])
    plt.plot(prec, '-')
    plt.plot(rec, '-')
    plt.plot(f1, '-')
    red_patch = mpatches.Patch(color='red', label='Precision')
    blue_patch = mpatches.Patch(color='blue', label='Recall')
    green_patch = mpatches.Patch(color='green', label='F1')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    #plt.show()
    plt.savefig('metrics'+name+'.png')

def plot_ROC(vec_R1, vec_FPR1, vec_R2, vec_FPR2, vec_R3, vec_FPR3):

    AUC1 = metrics.auc(vec_FPR1, vec_R1)
    AUC_str1 = str(AUC1)
    AUC2 = metrics.auc(vec_FPR2, vec_R2)
    AUC_str2 = str(AUC2)
    AUC3 = metrics.auc(vec_FPR3, vec_R3)
    AUC_str3 = str(AUC3)
    plt.clf()
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.plot(vec_FPR1, vec_R1, '-')
    plt.plot(vec_FPR2, vec_R2, '-')
    plt.plot(vec_FPR3, vec_R3, '-')
    AUC_value1 = mpatches.Patch(color='blue', label='AUC = ' + AUC_str1)
    AUC_value2 = mpatches.Patch(color='blue', label='AUC = ' + AUC_str2)
    AUC_value3 = mpatches.Patch(color='blue', label='AUC = ' + AUC_str3)
    plt.legend(handles=[AUC_value1, AUC_value2, AUC_value3])
    plt.savefig('roc_overall.png')
    #plt.show()



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

    return vec_FP, vec_FN, vec_TP, vec_TN,


def accumulate_values(vec_FP, vec_FN, vec_TP, vec_TN, vec_P, vec_R, vec_F1, vec_FPR, AccFP, AccFN, AccTP, AccTN):

    """
    Description: accumulate values
    Input: AccFP, AccFN, AccTP, AccTN, AccP, AccR, AccF1
    Output: vec_FP, vec_FN, vec_TP, vec_TN, vec_P, vec_R, vec_F1
    """

    vec_FP.append(AccFP)
    vec_FN.append(AccFN)
    vec_TP.append(AccTP)
    vec_TN.append(AccTN)
    FP = AccFP
    FN = AccFN
    TP = AccTP
    TN = AccTN
    if float(TP + FP) != 0.0:
        P = TP / float(TP + FP)
    else:
        P = 0
        # Recall (R)
    if float(TP + FN) != 0.0:
        R = TP / float(TP + FN)
    else:
        R = 0
        # F1 score (F1)
    if float(P + R) != 0.0:
        F1 = 2 * P * R / (P + R)
    else:
        F1 = 0
    if float(FP + TN) != 0.0:
        FPR = FP / float(FP + TN)
    else:
        FPR = 0

    vec_P.append(P)
    vec_R.append(R)
    vec_F1.append(F1)
    vec_FPR.append(FPR)
    return vec_FP, vec_FN, vec_TP, vec_TN, vec_P, vec_R, vec_F1, vec_FPR


def get_metrics(TP, TN, FP, FN):
    if float(TP + FP) != 0.0:
        P = TP / float(TP + FP)
    else:
        P = 0
        # Recall (R)
    if float(TP + FN) != 0.0:
        R = TP / float(TP + FN)
    else:
        R = 0
        # F1 score (F1)
    if float(P + R) != 0.0:
        F1 = 2 * P * R / (P + R)
    else:
        F1 = 0
    if float(FP + TN) != 0.0:
        FPR = FP / float(FP + TN)
    else:
        FPR = 0
    return P, R, F1, FPR