#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import cv2
import numpy as np
from skimage.measure import label
from skimage.measure import regionprops


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
    if path_test == "./highway/input/":
        accumulator = np.zeros((240,320,150), np.float32)
    if path_test == "./traffic/input/":
        accumulator = np.zeros((240,320,50), np.float32)

    return accumulator


def track_blobs(frame, background_filtered, minAreaPixels):

    """
    Description: track blobs
    Input: frame, background_filtered, minAreaPixels
    Output: frame
    """

    # Copy background subtraction to new image to track
    background_tracker = background_filtered.copy() 
    background_tracker = cv2.cvtColor(background_tracker, cv2.COLOR_GRAY2RGB)            
    # Area filltering, label background regions
    label_image = label(background_filtered)
    # Measure properties of labeled background regions
    for region in regionprops(label_image):
        # Remove regions smaller than fixed area
        if region.area > minAreaPixels:
            minr, minc,  maxr, maxc = region.bbox
            # Print rectangle of bounding box
            cv2.rectangle(frame, (minc,minr), (maxc,maxr), (0,255,0), 2)
            # Print centroid of bounding box	
            cv2.circle(frame, (minc+((maxc-minc)/2),minr+((maxr-minr)/2)), 3, (0,0,255), -1)

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


