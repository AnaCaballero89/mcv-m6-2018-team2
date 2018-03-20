#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import sys
import math
import cv2
import numpy as np


def get_statistics(resultOF, gtOF):
    
    """
    Description: compute statistics
    Input: 
    Output: 
    """

    errorVector = []
    correctPrediction = []
    uResult = []
    vResult = []
    uGT = []
    vGT = []
    imageToReconstruct = []
    validGroundTruth = []

    for pixel in range(0,resultOF[:,:,0].size):
        uResult.append( ((float)(resultOF[:,:,0].flat[pixel]) - math.pow(2, 15) ) / 64.0 )
        vResult.append(((float)(resultOF[:,:,1].flat[pixel])-math.pow(2, 15))/64.0)
        uGT.append(((float)(gtOF[:,:,0].flat[pixel])-math.pow(2, 15))/64.0)
        vGT.append(((float)(gtOF[:,:,1].flat[pixel])-math.pow(2, 15))/64.0)

    for idx in range(len(uResult)):
        squareError = math.sqrt(math.pow((uGT[idx] - uResult[idx]), 2) + math.pow((vGT[idx] - vResult[idx]), 2))
        errorVector.append(squareError)
        imageToReconstruct.append(squareError)
        if (squareError > 3):
            correctPrediction.append(0)
        else:
            correctPrediction.append(1)

    error = (1 - sum(correctPrediction)/(float)(len(uResult))) * 100;
    errorArray = np.asarray(errorVector)
    return errorArray, error


