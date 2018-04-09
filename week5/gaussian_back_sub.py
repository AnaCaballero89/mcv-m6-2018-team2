#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import math
import cv2
import pymorph as pym
import numpy as np
from scipy import ndimage
#from evaluate import *
#from sklearn.metrics import confusion_matrix
#from skimage.segmentation import clear_border
#from PIL import Image
from skimage.measure import label
from skimage.measure import regionprops
#from util import preprocess_pred_gt
from morphology import *


# Define colors spaces to transform frames
colorSpaceConversion={}
colorSpaceConversion['YCrCb'] = cv2.COLOR_BGR2YCR_CB
colorSpaceConversion['HSV']   = cv2.COLOR_BGR2HSV
colorSpaceConversion['gray'] = cv2.COLOR_BGR2GRAY

# Path to save images and videos
images_path = "std-mean-images/"
video_path = "background-subtraction-videos/"

# Define groundtruth labels namely
STATIC = 0
HARD_SHADOW = 50
OUTSIDE_REGION = 85
UNKNOW_MOTION = 170
MOTION = 255


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
    if path_test == "/imatge/froldan/work/highway/input/":
        accumulator = np.zeros((240,320,150), np.float32)
    if path_test == "/imatge/froldan/work/fall/input/":
        accumulator = np.zeros((480,720,50), np.float32)
    if path_test == "/imatge/froldan/work/traffic/input/":
        accumulator = np.zeros((240,320,50), np.float32)

    return accumulator


def gaussian_color(frame, mu_matrix, sigma_matrix, alpha, colorSpace, connectivity, areaPixels,ac_morphology,SE1size,SE2size):


    # Define the codec and create VideoWriter object
    # Define structuring element according to connectivity
    structuring_element = [[0,0,0],[0,0,0],[0,0,0]]
    if connectivity == '4':
        structuring_element = [[0,1,0],[1,1,1],[0,1,0]]
    if connectivity == '8':
        structuring_element = [[1,1,1],[1,1,1],[1,1,1]]

    if colorSpace != 'RGB':
        frame = cv2.cvtColor(frame, colorSpaceConversion[colorSpace])

    background = np.prod(abs(frame - mu_matrix) >= alpha*(sigma_matrix+2),axis=2)

    background = background.astype(int)

    background = ndimage.binary_fill_holes(background, structure=structuring_element).astype(int)

    if ac_morphology==1:
        background = dilation(background,SE1size)
        background = ndimage.binary_fill_holes(background, structure=structuring_element).astype(int)
        background = erosion(background,SE1size)
        background = remove_dots(background,SE2size)

    # Replace 1 by 255
    background[background == 1] = 255
    # Scales, calculates absolute values, and converts the result to 8-bit
    background = cv2.convertScaleAbs(background)

    # Area filltering, label background regions
    label_image = label(background)
    # Measure properties of labeled background regions
    if areaPixels > 0:
        for region in regionprops(label_image):
            # Remove regions smaller than fixed area
            if region.area < areaPixels:
                minr, minc,  maxr, maxc = region.bbox
                background[minr:maxr,minc:maxc] = 0


    return background


