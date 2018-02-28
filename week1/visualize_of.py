import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from skimage.transform import resize

results1= './results/kitti/LKflow_000045_10.png'
results = cv2.imread(results1, -1)
image = './results/kitti/000045_10.png'
image = cv2.imread(image, -1)

u = []
v = []

results = block_reduce(results, block_size=(8, 8, 1), func=np.mean)
c1, c2, c3 = results.shape

x, y = np.meshgrid(np.arange(0, c2, 1), np.arange(0, c1, 1))
results_u = [(((float)(results[:, :, 1].flat[pixel]) - math.pow(2, 15)) / 64.0)/200.0 if results[:, :, 0].flat[pixel] == 1 else 0
     for pixel in range(0, results[:, :, 0].size)]
results_v = [(((float)(results[:, :, 2].flat[pixel]) - math.pow(2, 15)) / 64.0)/200.0 if results[:, :, 0].flat[pixel] == 1 else 0
     for pixel in range(0, results[:, :, 0].size)]

u = np.reshape(results_u, (c1, c2))
v = np.reshape(results_v, (c1, c2))
img = resize(image, (c1, c2))

plt.imshow(img)
Q = plt.quiver(x, y, u, v, pivot='mid', units='inches', scale_units='inches')
plt.show(Q)