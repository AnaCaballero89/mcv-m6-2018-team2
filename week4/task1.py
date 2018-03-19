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

# Sequences optical flow
training_path = "dataset/testing/"
testing_path = "dataset/training/"
# Groundtruth of non-occulded areas
gt_noc_path = "dataset/gt/flow_noc"
# Grountruth of occulsion regions
gt_occ_path = "dataset/gt/flow_occ"


if __name__ == "__main__":
   
    # Get sequence from directory 
    sequence = [] 
    for filename in sorted(os.listdir(training_path)):
        sequence.append(filename)    

    # Read frames until sequence is not empty
    while sequence:
 
         # Get firs frame
         filename_1 = sequence.pop(0)
         frame_1 = cv2.imread(training_path+filename_1)
         frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)

         # Get second frame
         filename_2 = sequence.pop(0)
         frame_2 = cv2.imread(training_path+filename_2)
         frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

         # Compute block matching
         print "Computing block matching ["+str(filename_1)+", "+str(filename_2)+"] ..."
         motion_x, motion_y = block_matching(frame_1, frame_2)
         print "Computing block matching ["+str(filename_1)+", "+str(filename_2)+"] ... done"     

         # Reshape frame to plot with motion
         frame_1 = np.array(frame_1[:,1:], dtype='float32')
         frame_2 = np.array(frame_2[:,1:], dtype='float32')

         # Plot motion 
         plot_motion(frame_1, frame_2, motion_x, motion_y)


