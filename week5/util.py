#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import cv2
import math
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops

# Dictionary of tracker positions using sort
tracker_dict = {}

# Dictionray of tracker poistions using meanshift
tracker_dict_meanshift = {}


def compute_meanshit(trackers, frame):

    """
    Description: compute meanshift
    Input: trackers, frame
    Output: none
    """

    # Iterate trackers
    for i in range(len(trackers)):

        # Get id from tracker
        key = int(trackers[i][4])
 
        # If dection was not process on meanshift
        if not key in tracker_dict_meanshift:

            # Setup initial location of window
            c = int(trackers[i][0])
            r = int(trackers[i][1])
            w = int(abs(trackers[i][2]-trackers[i][0]))
            h = int(abs(trackers[i][3]-trackers[i][1]))

            # Check windows is valid
            if c>0 and r>0 and w>0 and h>0:

                # Set up the ROI for tracking
                track_window = (c,r,w,h)
                roi = frame[r:r+h, c:c+w]
                hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

                # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

                # Save properties to compute mean shift on next iterations
                vec = []
                vec.append(track_window)
                vec.append(term_crit)
                vec.append(roi_hist)
                tracker_dict_meanshift[key] = vec



def predict_meanshit(trackers, frame):

    """
    Description: predict by meanshift
    Input: trackers, frame
    Output: none
    """

    # Get actives trackers
    current_trackers = []      
    for i in range(len(trackers)):
         key = int(trackers[i][4])
         current_trackers.append(key)

    # Iterate tracker ditc meanshift
    for i in tracker_dict_meanshift:

        if i in current_trackers:
            # Get properties to compute mean shift
            vec = tracker_dict_meanshift[i]  
            roi_hist = vec[2]
            term_crit = vec[1]
            track_window = vec[0]

            # Apply meanshift to get the new location
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)

            # Update track window
            vec[0] = track_window
            tracker_dict_meanshift[i] = vec

            # Draw it on image
            x,y,w,h = track_window
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)


def display_motion(frame):

    """
    Description: display motion
    Input: frame
    Output: none
    """

    # Itearte dictionary of tracker positions
    for key, value in tracker_dict.iteritems():
        # Itearte vector of positions
        for i in range(len(value)-1): 
            # Draw line to display motion
            cv2.line(frame,value[i],value[i+1],(0,0,255),2)

    return frame


def save_tracker_positions(trackers):

    """
    Description: save tracker positions
    Input: frame, trackers
    Output: none
    """

    # Iterate trackers
    for i in range(len(trackers)):
        
        # Get id from tracker
        key = int(trackers[i][4])
        
        # Get centroid coordinates       
        x = int(trackers[i][0]+((trackers[i][2]-trackers[i][0])/2))
        y = int(trackers[i][1]+((trackers[i][3]-trackers[i][1])/2))

        # Check if exists tracker in dictionary
        if key in tracker_dict:
            vec = tracker_dict[key]
            vec.append((x,y))
            tracker_dict[key] = vec

        # Insert new key in dictionary
        else:
            vec = []
            vec.append((x,y))
            tracker_dict[key] = vec
        


def get_accumulator(path_test):

    """
    Description: get accumulator structure data
    Depends on image size to define borders
    Data are coded into 32 bits of floats
    Input: path test 
    Output: accumulator
    """

    # Initialize accumualtor
    accumulator = np.zeros((0,0), np.float32) 

    # Set accumulator depending on dataset choosen
    if path_test == "./dataset/highway/input/":
        accumulator = np.zeros((240,320,150), np.float32)
    if path_test == "./dataset/traffic/input/":
        accumulator = np.zeros((240,320,50), np.float32)

    return accumulator


def get_centroids(foreground, minAreaPixels):

    """
    Description: track blobs
    Input: foreground, minAreaPixels
    Output: centroids
    """

    # Initialize centroids
    centroids = []

    # Copy foreground subtraction to new image to track
    foreground_tmp = foreground.copy() 
    foreground_tmp = cv2.cvtColor(foreground, cv2.COLOR_GRAY2RGB)       
     
    # Area filltering, label foreground regions
    label_image = label(foreground)

    # Measure properties of labeled foreground regions
    for region in regionprops(label_image):

        # Remove regions smaller than fixed area
        if region.area > minAreaPixels:

            # Get centroid position from region box
            minr, minc,  maxr, maxc = region.bbox
            x = minc+((maxc-minc)/2)
            y = minr+((maxr-minr)/2)
            centroids.append([minc,minr,maxc,maxr])

    return centroids


def display_detections(frame, foreground, minAreaPixels):

    """
    Description: track blobs
    Input: foreground, minAreaPixels
    Output:
    """

    # Copy foreground subtraction to new image to track
    foreground_tmp = foreground.copy() 
    foreground_tmp = cv2.cvtColor(foreground, cv2.COLOR_GRAY2RGB)       
     
    # Area filltering, label foreground regions
    label_image = label(foreground)

    # Measure properties of labeled foreground regions
    for region in regionprops(label_image):

        # Remove regions smaller than fixed area
        if region.area > minAreaPixels:

            # Get centroid position
            minr, minc,  maxr, maxc = region.bbox
            x = minc+((maxc-minc)/2)
            y = minr+((maxr-minr)/2)

            # Print rectangle of bounding box
            cv2.rectangle(frame, (minc,minr), (maxc,maxr), (0,255,0), 2)
            # Print centroid of bounding box	
            cv2.circle(frame, (x, y), 3, (0,0,255), -1)

    return frame


def area_filtering(background, minAreaPixels):

    """
    Description: area filtering
    Input: background, minAreaPixels
    Output: background filtered
    """

    # Initialize background subtractor filtered
    background_filtered = np.zeros((background.shape[0], background.shape[1]), np.float32) 

    # Area filltering, label background regions
    label_image = label(background)
    # Measure properties of labeled background regions
    for region in regionprops(label_image):
        # Remove regions smaller than fixed area
        if region.area > minAreaPixels:
            minr, minc,  maxr, maxc = region.bbox
            background_filtered[minr:maxr, minc:maxc] = background[minr:maxr, minc:maxc] 

    return background_filtered


