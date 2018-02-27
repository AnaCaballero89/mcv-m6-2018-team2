import cv2
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt


def msen(opticalflowResults, opticalflowGT, original):
    # opticalflowResults: Results from our optical flow calculation
    # opticalflowGT: Results of the optical flow from the ground truth.

    error_count = 0
    error = []

    results = cv2.imread(opticalflowResults, -1)
    opticalflow = cv2.imread(opticalflowGT, -1)
    original_image = cv2.imread(original, -1)

    # flow_u(u, v) = ((float)I(u, v, 1) - 2 ^ 15) / 64.0;
    # flow_v(u, v) = ((float) I(u, v, 2) - 2 ^ 15) / 64.0;
    # valid(u, v) = (bool)I(u, v, 3);

    results_u1 = results[:, :, 1].flatten().astype(np.float)
    results_v1 = results[:, :, 2].flatten().astype(np.float)
    results_validation = results[:, :, 0].flatten()

    results_u = [(pred - math.pow(2, 15)) / 64.0 for pred in results_u1]
    results_v = [(pred - math.pow(2, 15)) / 64.0 for pred in results_v1]

    groundtruth_u1 = opticalflow[:, :, 1].flatten().astype(np.float)
    groundtruth_v1 = opticalflow[:, :, 2].flatten().astype(np.float)
    groundtruth_validation = opticalflow[:, :, 0].flatten()

    groundtruth_u = [(gt - math.pow(2, 15)) / 64.0 for gt in groundtruth_u1]
    groundtruth_v = [(gt - math.pow(2, 15)) / 64.0 for gt in groundtruth_v1]


    #error = [gt - pred for gt, pred in zip(opticalflow, results)]

    error = np.zeros(len(groundtruth_u))
    error = []
    image_validated = []
    correct_map = []
    for i in range(len(groundtruth_u)):
        if groundtruth_validation[i] == 0:
            image_validated.append(0)
            continue
        else:
            errorpixels = math.sqrt(math.pow((groundtruth_u[i] - results_u[i]), 2) + math.pow((groundtruth_v[i] - results_v[i]), 2))
            error.append(errorpixels)
            image_validated.append(errorpixels)

        if errorpixels > 3:
            correct_map.append(0)
        else:
            correct_map.append(1)

    pepn = (1 - sum(correct_map)/(float)(sum(groundtruth_validation))) * 100
    mse = (1 / len(error)) * sum(error)


    print('Mean square error in Non-Occluded areas: {} '.format(mse))
    print('Percentage of Erroneous Pixels in Non-Occluded areas: {} % '.format(pepn))

    representation_OF(error)


    colormap_OF(image_validated,original_image)

def representation_OF(error):
    num_bins = 100
    n, bins, patches = plt.hist(error, num_bins, normed=1, facecolor='blue')
    plt.xlabel('Mean Square Error in Non-Occluded areas')
    plt.ylabel('Percentage of errors')
    plt.title(r'Histogram of errors')
    plt.show()

def colormap_OF(error_vector,originalimage):
    r, c= originalimage.shape
    error_map = np.reshape(error_vector, (r, c))

    plt.imshow(error_map)
    plt.colorbar()
    plt.show()



if __name__ == "__main__":
    results_1 = '/home/guillem/results_opticalflow_kitti/results/LKflow_000045_10.png'
    groundtruth_1 = '/home/guillem/data_stereo_flow/training/flow_noc/000045_10.png'
    original_1 = '/home/guillem/data_stereo_flow/training/image_0/000045_10.png'
    results_2 = '/home/guillem/results_opticalflow_kitti/results/LKflow_000157_10.png'
    groundtruth_2 = '/home/guillem/data_stereo_flow/training/flow_noc/000157_10.png'
    original_2 = '/home/guillem/data_stereo_flow/training/image_0/000157_10.png'
    msen(results_1, groundtruth_1, original_1)
    msen(results_2, groundtruth_2, original_2)