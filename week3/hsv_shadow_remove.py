import cv2
import numpy as np

def hsv_shadow_remove(frame_rgb, background_rgb):
    #Define variable
    alpha = 0.4
    beta = 0.6
    th = 0.1
    ts = 0.5
    #CHANGE THE INITIALIZATION
    foreground_without_shadows = []

    hsv_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)
    hsv_background = cv2.cvtColor(background_rgb, cv2.COLOR_BGR2HSV)

    #Conditions to fulfill
    #Pixel is considered shadow if it fulfills these 3 conditions
    cond1 = np.divide(hsv_frame[:, :, 2], hsv_background[:, :, 2])
    condition1 = (cond1 >= alpha)
    condition2 = (cond1 <= beta)
    condition3 = (hsv_frame[:, :, 1]-hsv_background[:, :, 1] <= ts)
    condition4 = (hsv_frame[:, :, 0]-hsv_background[:, :, 0] <= th)

    shadow_mask = np.bitwise_and(np.bitwise_and(condition1, condition2), np.bitwise_and(condition3, condition4))

    # Return Boolean mask
    return shadow_mask