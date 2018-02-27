import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.axes as mplax


u = []
v = []
results1= './results/kitti/LKflow_000045_10.png'
image = './results/kitti/000045_10.png'
results = cv2.imread(results1, -1)
c1, c2, c3 = results.shape
print c1
print c2
print c3
x, y = np.meshgrid(np.arange(0, c1, 1), np.arange(0, c2, 1))
#x, y = np.meshgrid(np.arange(0, 2 * np.pi, 1), np.arange(0, 2 * np.pi, 1))
results_u1 = results[:, :, 1].flatten().astype(np.float)
results_v1 = results[:, :, 2].flatten().astype(np.float)
results_validation = results[:, :, 0].flatten()
results_u = [(pred - math.pow(2, 15)) / 64.0 for pred in results_u1]
results_v = [(pred - math.pow(2, 15)) / 64.0 for pred in results_v1]
u = np.reshape(results_u, (c1, c2))
v = np.reshape(results_v, (c1, c2))
plt.figure()
plt.quiver(x, y, u, v, units='inches', scale=1, hatch=' ', alpha = 0.3, linewidth = 0.01)
plt.savefig('of_rep.png')