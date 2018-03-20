#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import sys
import cv2
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.measure import block_reduce
from matplotlib.ticker import FuncFormatter


def to_percent(y, position):

    """
    Description: to percentantge
    Input: y, position
    Output: none
    """

    s = str(100 * y)
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def plot_histogram(msen, error, filename):

    """
    Description: plot histogram
    Input: msen, error, filename
    Output: none
    """

    plt.hist(msen, bins = 25, normed = True)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xlabel('MSEN value')
    plt.ylabel('Number of Pixels')
    plt.title("Histogram of scene %s. \n Percentage of Erroneous Pixels in Non-occluded areas (PEPN): %d %%" % (filename, error))
    plt.show()


def plot_motion(frame, motion, N):

    """
    Description: Plot motion vector field
    Input: frame, motion, N
    Output: none
    """

    u = []
    v = []
    c1, c2, c3 = motion.shape
    motion = block_reduce(motion, block_size=(N, N, 1), func=np.mean)
    c1, c2, c3 = motion.shape
    x, y = np.meshgrid(np.arange(0, c2, 1), np.arange(0, c1, 1))
    motion_u = [(((float)(motion[:, :, 0].flat[pixel]) - math.pow(2, 15)) / 64.0)/200.0 if motion[:, :, 0].flat[pixel] == 1 else 0
    for pixel in range(0, motion[:, :, 0].size)]
    motion_v = [(((float)(motion[:, :, 1].flat[pixel]) - math.pow(2, 15)) / 64.0)/200.0 if motion[:, :, 0].flat[pixel] == 1 else 0
    for pixel in range(0, motion[:, :, 0].size)]
    u = np.reshape(motion_u, (c1, c2))
    v = np.reshape(motion_v, (c1, c2))
    img = resize(frame, (c1, c2))
    plt.imshow(img)
    Q = plt.quiver(x, y, u, v, pivot='mid', units='inches', scale_units='inches')
    plt.show(Q)


