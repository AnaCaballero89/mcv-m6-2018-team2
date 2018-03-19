#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import sys
import cv2
import numpy as np

# Block size (N)
# Large enough to have statistically relevant data 
# Small enough to represent part of an object under translation. 
# Typically 16x16 pixels.
N = 16

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


def reshape_motion(motion):

    """
    Description: reshape motion matrix
    Input: motion
    Output: new motion
    """

    # Define new motion 
    new_motion = np.zeros([motion.shape[0]*N,motion.shape[1]*N])

    # Iterate motion to get new motion
    for x in range(motion.shape[0]):
        for y in range(motion.shape[1]):
            new_motion[x*N:x*N+N, y*N:y*N+N] = motion[x,y] 

    return new_motion


def compute_motion(region, block):

    """
    Description: compute motion from region depending
    on block size
    Input: region, block
    Output: motion x and motion
    """

    # Get rows and columns
    [rows, columns] = region.shape

    # Get maximum representable finite float
    diff = sys.float_info.max

    # Iterate region 
    for x in range(rows-P):
        for y in range(columns-P):

            # Get block from region
            current_block = region[x:x+N, y:y+N]
           
            # Compute difference between blocks
            current_diff = sum(sum(abs(current_block-block)**2))

            # Update difference 
            if current_diff < diff:
                diff = current_diff
                x_motion = - x + P
                y_motion = y - P

    return x_motion, y_motion


def block_matching(frame_1, frame_2):

    """
    Description: block matching
    Input: frame_1, frame_2
    Output: motion x and motion y
    """

    # Set motion compensation
    if MC == 'backward':
        img_1 = frame_2
        img_2 = frame_1
    if MC == 'forward':
        img_1 = frame_1
        img_2 = frame_2
    
    # Get rows and columns by blocks 
    [rows, columns] = img_1.shape
    rows = rows / N
    columns = columns / N

    # Add padding in the search image
    img_2_padd = np.zeros([img_1.shape[0]+2*P,img_1.shape[1]+2*P])
    img_2_padd[P:P+img_1.shape[0],P:P+img_1.shape[1]] = img_2[:,:]

    # Set motion matrices depending on block size
    motion_x = np.zeros([rows, columns])
    motion_y = np.zeros([rows, columns])
    
    # Iterate frame by blocks in both axis
    for x in range(rows):
        for y in range(columns):

            # Get block depending on block size (N)
            block = img_1[x*N:x*N+N, y*N:y*N+N]
       
            # Get region depending on area search (P)
            region = img_2_padd[x*N:x*N+N+2*P, y*N:y*N+N+2*P]

            # Compute motion
            x_mot, y_mot = compute_motion(region, block)
            
            # Update motion matrices
            motion_x[x, y] = x_mot
            motion_y[x, y] = y_mot
            
    # Cast to float32 type
    motion_x = motion_x.astype('float32')
    motion_y = motion_y.astype('float32')

    # Reshape motion x and y
    motion_x = reshape_motion(motion_x)
    motion_y = reshape_motion(motion_y)

    return motion_x, motion_y


