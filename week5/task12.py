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
detrac_path_in = "./dataset/detrac/"
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

    # W5 T1.2 Tracking with MeanShift prediction
    # Structure of Kalman filter is used to get new object detections
    # Apply the background substraction work previously done

    # Choose betwen: highway, traffic or detrac
    path_test, first_frame, last_frame = setup("detrac")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path+"mean_shift_"+str(path_test.split("/")[2])+".avi", fourcc, 60, (get_accumulator(path_test).shape[1], get_accumulator(path_test).shape[0]))

    # Read sequence of images sorted
    for filename in sorted(os.listdir(path_test)):
       
        # Display current frame
        print "Processing frame: "+str(filename)

        # Check that frame is into range
        frame_num = int(filename[3:8])
        if frame_num >= first_frame and frame_num <= last_frame:

            # Read image from groundtruth 
            frame = cv2.imread(path_test+filename)

            # Apply background subraction
            background = fgbg.apply(frame)
            background = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel)
           
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
            save_tracker_positions(trackers)

            # Predict by menashift
            compute_meanshit(trackers, frame)
            predict_meanshit(trackers, frame)

            # Show results
            cv2.imshow("tracking MEANSHIFT", frame)
            cv2.waitKey(1)

            # Write frame into video
            out.write(np.uint8(frame))

            # Update frame_counter
            frame_counter = frame_counter + 1

         
