#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import numpy as np
from util import *
from sort import *
from scipy import ndimage
#from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

# Path configurations
highway_path_in = "./dataset/highway/input/"
traffic_path_in = "./dataset/traffic/input/"
video_path = "./videos/"

# Connectivity 8 to hole filling
structuring_element = [[1,1,1],[1,1,1],[1,1,1]]

# Set area pixels to filter blobs
minAreaPixels = 100

# Set kernel to apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

# Set background subtraction from opencv
# It is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize centroids
centroids = []

# Create instance of the SORT tracker
# Simple Online and Realtime Tracking
tracker_motion = Sort() 

# Frame counter
frame_counter = 0


if __name__ == "__main__":

    # W5 T1.1 Tracking with Kalman filter using SORT tracker
    # Use Kalman filter to track each vehicle appearing in the sequence
    # Apply the background substraction work previously done

    # Set up configuration
    path_test = highway_path_in
    first_frame = 1050
    last_frame = 1350

    # path_test = traffic_path_in
    # first_frame = 950
    # last_frame = 1050

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path+"kalman_filter_"+str(path_test.split("/")[2])+".avi", fourcc, 60, (get_accumulator(path_test).shape[1], get_accumulator(path_test).shape[0]))
    out_bg = cv2.VideoWriter(video_path+"bg_"+str(path_test.split("/")[2])+".avi", fourcc, 60, (get_accumulator(path_test).shape[1], get_accumulator(path_test).shape[0]))

    # Read sequence of images sorted
    for filename in sorted(os.listdir(path_test)):
       
        # Check that frame is into range
        frame_num = int(filename[2:8])
        if frame_num >= first_frame and frame_num <= last_frame:

            # Read image from groundtruth 
            frame = cv2.imread(path_test+filename)

            # Apply background subraction
            background = fgbg.apply(frame)
           
            # Filter detections by area
            background_filtered = area_filtering(background, minAreaPixels)

            # Apply hole filling
            background_filtered = ndimage.binary_fill_holes(background_filtered, structure=structuring_element).astype(float)
            background_filtered = np.float32(background_filtered)

            # Track blobs
            centroids = get_centroids(background_filtered, minAreaPixels)

            # Update tracker motion
            dets = np.array(centroids)
            trackers = tracker_motion.update(dets)

            # Save tracker values
            save_tracker_positions(trackers)
            frame = display_motion(frame)

            # Show results
            cv2.imshow("background subtraction", background_filtered)
            frame = display_detections(frame, background_filtered, minAreaPixels)
            cv2.imshow("tracking KALMAN FILTER", frame)
            cv2.waitKey(1)

            # Write frame into video
            background_filtered[background_filtered == 1] = 255
            background_filtered = cv2.convertScaleAbs(background_filtered)
            background_filtered = cv2.cvtColor(background_filtered, cv2.COLOR_GRAY2RGB)
            out_bg.write(background_filtered)
            out.write(np.uint8(frame))

            # Update frame_counter
            frame_counter = frame_counter + 1

         
