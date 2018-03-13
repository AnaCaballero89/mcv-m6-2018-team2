from gaussian_color import *
from train_color import *
from util import *
from sklearn import metrics
"""
THIS SCRIPT ONLY HANDLES FROM TASK1 AND TASK2. ADD MORPHOLOGY OPTIONS TO SUPPORT CURRENT VERSION.
"""
# Highway sequences configuration, range 1050 - 1350
highway_path_in = "./highway/input/"
highway_path_gt = "./highway/groundtruth/"

# Fall sequences configuration, range 1460 - 1560
fall_path_in = "./fall/input/"
fall_path_gt = "./fall/groundtruth/"

# Traffic sequences configuration, range 950 - 1050
traffic_path_in = "./traffic/input/"
traffic_path_gt = "./traffic/groundtruth/"

# Group sequences
path_tests = [highway_path_in,fall_path_in,traffic_path_in]
path_gts = [highway_path_gt,fall_path_gt,traffic_path_gt]
first_frames = [1050,1460,950]
middle_frames = [1199,1509,999]
last_frames = [1349,1559,1049]
num_dataset = 2
# Define color spaces ['RGB','HSV','YCrCb']
colorSpaces=['YCrCb', 'YCrCb', 'RGB']
alpha=0.5
connectivity = '4'
minAreaP = 0
print 'Starting...'
mean_matrix, std_matrix = training_color(path_tests[num_dataset], first_frames[num_dataset], middle_frames[num_dataset],
                                         alpha, 'YCrCb')
FP, FN, TP, TN, P, R, F1 = gaussian_color(path_tests[num_dataset], path_gts[num_dataset],
                                                           middle_frames[num_dataset] + 1, last_frames[num_dataset],
                                                           mean_matrix, std_matrix, alpha, 'YCrCb', connectivity,
                                                           minAreaP)
print 'Finished!!!'
