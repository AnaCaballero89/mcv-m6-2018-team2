import cv2
import numpy as np
import math
import os


dataset = '/home/fran/Downloads/training/image_0/'
gt ='/home/fran/Downloads/training/flow_noc/000157_10.png'
groundtruth = cv2.imread(gt, -1)
groundtruth_validation = groundtruth[:, :, 0].flatten()
image_pair = ['000157_10.png', '000157_11.png'] #Change image pair names for different tests
frame1 = cv2.imread(os.path.join(dataset, image_pair[0]), cv2.IMREAD_COLOR)
frame2 = cv2.imread(os.path.join(dataset, image_pair[1]), cv2.IMREAD_COLOR)
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
cv2.imwrite('opticalfb_'+image_pair[0]+'.png',frame2)
cv2.imwrite('opticalhsv_'+image_pair[0]+'.png',bgr)

prvs = next
cv2.destroyAllWindows()
gt_x = groundtruth[0, :, :1]
gt_y = groundtruth[1, :, :1]
gt_mask = groundtruth[2, :, :1]
error_x = np.subtract(flow[0,:], gt_x)
error_y = np.subtract(flow[1,:], gt_y)
print(error_x.shape)
total_error = np.array([math.sqrt((x**2) + (y**2)) for x, y in zip(error_x.ravel(), error_y.ravel())])
msen = np.sum(total_error)/float(np.sum(total_error>0))
print(msen)
pepn = len([i for i,x in enumerate(total_error) if x>3])/float((sum(groundtruth_validation))) * 100
print(pepn)