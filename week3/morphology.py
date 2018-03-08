#!/usr/bin/env python
__author__  = "Master Computer Vision. Team 02"
__license__ = "M6 Video Analysis"

# Import libraries
import os
import cv2


def remove_dots(img, se_size):

    """
    Description: remove dots
    If se_size not specified, it is assumed to be in the center
    Input: img, se_size
    Output: img
    """

    # Create kernel as ellipse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, se_size))
    # Apply opening, obtained by the erosion of an image followed by a dilation
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img


def remove_vertical_lines(image, se_size):

    """
    Description: remove vertical lines
    If se_size not specified, it is assumed to be in the center
    Input: img, se_size
    Output: img
    """

    # Create kernel as rectangular box, with only one column
    # to adjust structuring element to vertical lines considered as noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, se_size))
    # Apply closing, obtained by the dilation of an image followed by an erosion
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


def remove_horizontal_lines(img, se_size):

    """
    Description: remove horizontal lines
    If se_size not specified, it is assumed to be in the center
    Input: img, se_size
    Output: img
    """

    # Create kernel as rectangular box, with only one row
    # to adjust structuring element to horizontal lines considered as noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, 1))
    # Apply closing, obtained by the dilation of an image followed by an erosion
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


