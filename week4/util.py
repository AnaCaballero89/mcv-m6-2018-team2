#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_motion(frame_1, frame_2, motion_x, motion_y):

    """
    Description: plot motion
    Input: frame_1, frame_2, motion_x, motion_y
    Output: none
    """

    # Plot motion vector field
    plt.title('Motion vector field with arrows')
    plt.quiver(frame_1, frame_2, motion_x, motion_y, edgecolor='k', facecolor='None', linewidth=0.25)
    plt.xlim(-1, 255)
    plt.xticks(())
    plt.ylim(-1, 255)
    plt.yticks(())
    plt.show()


