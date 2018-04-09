#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import numpy as np
from util import *
from sort import *
from gaussian_back_sub import *
from train_color import *
from scipy import ndimage
import os.path
import numpy as np

dataset="own2" # "highway","fall", "traffic", "traffic_stabilized", "own1", "own1_stabilized", "own2", "own2_stabilized"
# Path configurations
video_path = "./videos/"
# Initialize centroids
centroids = []
# Create instance of the SORT tracker (Simple Online and Realtime Tracking)
tracker_motion = Sort() 
# Frame counter
frame_counter = 0

path_in, first_train_frame, last_train_frame, first_test_frame, last_test_frame, im_size, alpha, colorSpace, connectivity, areaPixels, ac_morphology, SE1size, SE2size = setup(dataset)

if __name__ == "__main__":

    # W5 T1.1 Tracking with Kalman filter using SORT tracker
    # Use Kalman filter to track each vehicle appearing in the sequence
    # Apply the background substraction work previously done

    mu_matrix, sigma_matrix = training_color(path_in, first_train_frame, last_train_frame, alpha, colorSpace)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path+"kalman_filter_"+dataset+".avi", fourcc, 10, (im_size[1],im_size[0]))

    filenames=sorted(os.listdir(path_in))
    # Read sequence of images sorted
    for filename in filenames[first_test_frame:last_test_frame]:
       
        # Display current frame
        print "Processing frame: "+str(filename)

        # Read image from groundtruth
        frame = cv2.imread(path_in+filename)

        # Apply background subraction
        background_filtered = gaussian_color(frame, mu_matrix, sigma_matrix, alpha, colorSpace, connectivity, areaPixels,ac_morphology,SE1size,SE2size)
        cv2.imwrite("bk_subs_images/"+filename[0:(len(filename)-4)]+".jpg",np.uint8(background_filtered))

        # Track blobs
        centroids = get_centroids(background_filtered, areaPixels)

        # Update tracker motion
        dets = np.array(centroids)
        trackers = tracker_motion.update(dets)

        # Save tracker values
        save_tracker_positions(trackers)

        # Show results
        frame = display_detections(frame, background_filtered, areaPixels)
        frame = display_motion(frame, trackers)
        cv2.imshow("tracking KALMAN FILTER", frame)
        cv2.waitKey(1)
 
        # Write frame into video
        out.write(np.uint8(frame))

        # Update frame_counter
        frame_counter = frame_counter + 1

         
