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
    cond1 = np.divide(hsv_frame[:, :, 3], hsv_background[:, :, 3])
    shadow_mask = np.prod((abs(cond1 >= alpha))
                          and (abs(cond1 <= beta))
                          and (abs(hsv_frame[:, :, 2]-hsv_background[:, :, 2] <= ts))
                          and (abs(hsv_frame[:, :, 1]-hsv_background[:, :, 1] <= th))
                          )

    # Return Boolean mask
    return shadow_mask