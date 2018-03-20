#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import sys
import cv2
import numpy as np
from block_matching import *
from util import *
from evaluate import *

# Sequences optical flow
training_path = "dataset/testing/"
testing_path = "dataset/training/"
# Groundtruth of non-occulded areas
gt_noc_path = "dataset/gt/flow_noc/"
# Grountruth of occulsion regions
gt_occ_path = "dataset/gt/flow_occ/"

# Block size (N)
# Large enough to have statistically relevant data 
# Small enough to represent part of an object under translation. 
# Typically 16x16 pixels.
N = 8

# Search area (P)
# It is related to the range of expected movement:
# P pixels in every direction: (2P+N)x(2P+N) pixels. 
# Typically P = N.
P = N

# Quantization step (Q)
# Related to the accuracy of the estimated motion.
# Typically, 1 pixel but it can go down to 1/4 of pixel
# Need of interpolation in typically case
Q = 0

# Motion compensation (MC)
# - Forward motion estimation: all pixels in the past image are 
# associated to a pixel in the current image
# - Backward motion estimation: All pixels in the current image are 
# associated to a pixel in the past image
MC = 'backward'


if __name__ == "__main__":
   
    # Get sequence from directory 
    sequence = [] 
    for filename in sorted(os.listdir(training_path)):
        sequence.append(filename)    

    # Read frames until sequence is not empty
    while sequence:
 
         # Get firs frame in gray scale
         filename_1 = sequence.pop(0)
         frame_1 = cv2.imread(training_path+filename_1, -1)

         # Get second frame in gray scale
         filename_2 = sequence.pop(0)
         frame_2 = cv2.imread(training_path+filename_2, -1)
	
         # Compute block matching
         print "Computing block matching ["+str(filename_1)+", "+str(filename_2)+"] ..."
         motion = block_matching(frame_1, frame_2, N, P, MC, Q)
         print "Computing block matching ["+str(filename_1)+", "+str(filename_2)+"] ... done"     

         # Plot motion vector field
         plot_motion(frame_1, motion, N)
 
         # Load groundtruth image in gray scale and remove third dimension
         groundtruth = cv2.imread(gt_noc_path+filename_1, -1)
 
         # Reshape groundtruth and motion to compute statistics
         if groundtruth.shape[0] == 376 and groundtruth.shape[1] == 1241:
             groundtruth = groundtruth[:, 0:groundtruth.shape[1]-1, 0:2]
         if groundtruth.shape[0] == 375 and groundtruth.shape[1] == 1242:
             motion = motion[0:motion.shape[0]-1, :, 0:2]
             groundtruth = groundtruth[:, 0:groundtruth.shape[1]-2, 0:2]

         # Compute statistics 
         values, error = get_statistics(motion, groundtruth)

         # Plot histogram with MSEN values and PEPN
         plot_histogram(values, error, filename)

         

