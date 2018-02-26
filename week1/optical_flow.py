import cv2
import numpy as np
import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import json
import math
import matplotlib
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def msen(opticalflowResults, opticalflowGT):
    # opticalflowResults: Results from our optical flow calculation
    # opticalflowGT: Results of the optical flow from the ground truth.

    error_count = 0
    results = cv2.imread(opticalflowResults).flatten()
    opticalflow = cv2.imread(opticalflowGT).flatten()

    error = [gt - pred for gt, pred in zip(opticalflow, results)]
    mse = [x**2 for x in error]
    total_mse = (1/results.shape[0]) * sum(mse)

    print('Mean square error in Non-Occluded areas: {} '.format(total_mse))

    for e in error:
        if abs(e) > 3:
            error_count += 1

    pepn = 100 * (error_count / results.shape[0])
    print('Percentage of Erroneous Pixels in Non-Occluded areas: {} % '.format(pepn))

def representation_OF():


if __name__ == "__main__":
    results_1 = '/home/guillem/results_opticalflow_kitti/results/LKflow_000045_10.png'
    groundtruth_1 = '/home/guillem/data_stereo_flow/training/flow_noc/000045_10.png'
    results_2 = '/home/guillem/results_opticalflow_kitti/results/LKflow_000157_10.png'
    groundtruth_2 = '/home/guillem/data_stereo_flow/training/flow_noc/000157_10.png'
    msen(results_1, groundtruth_1)
    msen(results_2, groundtruth_2)