import cv2
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt


def msen(opticalflowResults, opticalflowGT):
    # opticalflowResults: Results from our optical flow calculation
    # opticalflowGT: Results of the optical flow from the ground truth.

    error_count = 0
    error = []

    results = cv2.imread(opticalflowResults, -1)
    opticalflow = cv2.imread(opticalflowGT, -1)

    # flow_u(u, v) = ((float)I(u, v, 1) - 2 ^ 15) / 64.0;
    # flow_v(u, v) = ((float) I(u, v, 2) - 2 ^ 15) / 64.0;
    # valid(u, v) = (bool)I(u, v, 3);

    results_u1 = results[:, :, 1].flatten().astype(np.float)
    results_v1 = results[:, :, 2].flatten().astype(np.float)
    results_validation = results[:, :, 0].flatten()

    results_u = [(pred - math.pow(2, 15)) / 64.0 for pred in results_u1]
    results_v = [(pred - math.pow(2, 15)) / 64.0 for pred in results_v1]

    for i in range(len(results_validation)):
        results_u[i] = results_u[i] * results_validation[i]
        results_v[i] = results_v[i] * results_validation[i]




    groundtruth_u1 = opticalflow[:, :, 1].flatten().astype(np.float)
    groundtruth_v1 = opticalflow[:, :, 2].flatten().astype(np.float)
    groundtruth_validation = opticalflow[:, :, 0].flatten()

    groundtruth_u = [(gt - math.pow(2, 15)) / 64.0 for gt in groundtruth_u1]
    groundtruth_v = [(gt - math.pow(2, 15)) / 64.0 for gt in groundtruth_v1]

    for i in range(len(groundtruth_validation)):
        groundtruth_u[i] = groundtruth_u[i] * groundtruth_validation[i]
        groundtruth_v[i] = groundtruth_v[i] * groundtruth_validation[i]


    #error = [gt - pred for gt, pred in zip(opticalflow, results)]

    error = np.zeros(len(groundtruth_u))
    for i in range(len(groundtruth_u)):
        error[i] = math.sqrt(math.pow((groundtruth_u[i] - results_u[i]), 2) + math.pow((groundtruth_v[i] - results_v[i]), 2))


    #mse = [x**2 for x in error]
    total_mse = (1/len(error)) * sum(error)

    print('Mean square error in Non-Occluded areas: {} '.format(total_mse))

    for e in error:
        if e > 3:
            error_count += 1

    pepn = 100 * (error_count / sum(groundtruth_validation))
    print('Percentage of Erroneous Pixels in Non-Occluded areas: {} % '.format(pepn))

    representation_OF(abs(error))

def representation_OF(error):
    num_bins = 100
    n, bins, patches = plt.hist(error, num_bins, normed=1, facecolor='blue')
    plt.xlabel('Mean Square Error in Non-Occluded areas')
    plt.ylabel('Percentage of errors')
    plt.title(r'Histogram of errors')
    plt.show()



if __name__ == "__main__":
    results_1 = '/home/guillem/results_opticalflow_kitti/results/LKflow_000045_10.png'
    groundtruth_1 = '/home/guillem/data_stereo_flow/training/flow_noc/000045_10.png'
    results_2 = '/home/guillem/results_opticalflow_kitti/results/LKflow_000157_10.png'
    groundtruth_2 = '/home/guillem/data_stereo_flow/training/flow_noc/000157_10.png'
    msen(results_1, groundtruth_1)
    msen(results_2, groundtruth_2)