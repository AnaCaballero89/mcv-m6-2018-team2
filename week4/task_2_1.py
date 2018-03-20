import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
#from block_matching import *

mode = 'folder'
ID = "Traffic"
path = "datasets/traffic/input/"
alpha = 1.9
rho = 0.03

data = sorted(glob.glob(path + "*.jpg"))
minFrame = 950
maxFrame = 1050
dataRange = np.arange(minFrame,maxFrame)

reference_frame = 950
reference_img = cv2.imread(data[reference_frame])
reference_img_BW = cv2.cvtColor(reference_img,cv2.COLOR_BGR2GRAY)

block_size = 16
area_size = 32
compensation = 'backward'  # or 'forward'

# Define the codec and create VideoWriter object
video_path = "stabilizated_videos/"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path+"task_2_1_"+".avi", fourcc, 20, (reference_img_BW.shape[0], reference_img_BW.shape[1]),1)



def block_search(region_to_explore, block_to_search):
    block_sizeX = block_to_search.shape[0]
    block_sizeY = block_to_search.shape[1]
    x_size = region_to_explore.shape[0]
    y_size = region_to_explore.shape[1]

    min_diff = sys.float_info.max
    x_mot = 0
    y_mot = 0
    for row in range(x_size-block_sizeX):
        for column in range(y_size-block_sizeY):
            block2analyse = region_to_explore[row:row+block_sizeX, column:column+block_sizeY]
            diff = sum(sum(abs(block2analyse-block_to_search)**2))
            if diff < min_diff:
                min_diff = diff
                x_mot = - row + area_size
                y_mot = column - area_size
    return x_mot, y_mot


# Auxiliary function
def compute_block_matching2017(prev_img, curr_img):

    img2xplore = curr_img
    searchimg = prev_img

    x_blocks = img2xplore.shape[0] / block_size
    y_blocks = img2xplore.shape[1] / block_size

    # Add padding in the search image
    pad_searchimg = np.zeros([img2xplore.shape[0] + 2 * area_size, img2xplore.shape[1] + 2 * area_size])
    pad_searchimg[area_size:area_size + img2xplore.shape[0], area_size:area_size + img2xplore.shape[1]] = searchimg[:,:]

    motion_matrix = np.zeros([x_blocks, y_blocks, 2])

    for row in range(x_blocks):
        for column in range(y_blocks):
            print "Computing block row_" + str(row) +"col_"+ str(column)
            block_to_search = img2xplore[row * block_size:row * block_size + block_size,
                              column * block_size:column * block_size + block_size]
            region_to_explore = pad_searchimg[row * block_size:row * block_size + block_size + 2 * area_size,
                                column * block_size:column * block_size + block_size + 2 * area_size]
            x_mot, y_mot = block_search(region_to_explore, block_to_search)

            motion_matrix[row, column, 0] = x_mot
            motion_matrix[row, column, 1] = y_mot

    return motion_matrix

def apply_stabilization(real_x, real_y, curr_img):
    x_size = curr_img.shape[0]
    y_size = curr_img.shape[1]

    x_blocks = x_size / block_size
    y_blocks = y_size / block_size

    if curr_img.shape.__len__() == 2:
        new_curr_img = np.zeros([x_size + 2 * area_size, y_size + 2 * area_size])
        new_curr_img[area_size:area_size + x_size, area_size:area_size + y_size] = curr_img
        comp_img = np.zeros([x_size, y_size])
    elif curr_img.shape.__len__() == 3:
        new_curr_img = np.zeros([x_size + 2 * area_size, y_size + 2 * area_size, 3])
        auxImg = np.pad(curr_img[:, :, 0], ((area_size, area_size), (area_size, area_size)), 'symmetric')
        new_curr_img[:, :, 0] = auxImg
        auxImg = np.pad(curr_img[:, :, 1], ((area_size, area_size), (area_size, area_size)), 'symmetric')
        new_curr_img[:, :, 1] = auxImg
        auxImg = np.pad(curr_img[:, :, 2], ((area_size, area_size), (area_size, area_size)), 'symmetric')
        new_curr_img[:, :, 2] = auxImg
        comp_img = np.zeros([x_size, y_size, 3])
    else:
        print 'ERROR dimension'
        return curr_img

    if curr_img.shape.__len__() == 2:
        comp_img[0:comp_img.shape[0], 0:comp_img.shape[1]] = new_curr_img[0+ real_x :comp_img.shape[0] + real_x , 0 + real_y:comp_img.shape[1]+ real_y]

    elif curr_img.shape.__len__() == 3:
        comp_img[0:comp_img.shape[0], 0:comp_img.shape[1], 0] = new_curr_img[area_size + real_x : area_size + comp_img.shape[0] + real_x , area_size + real_y : area_size + comp_img.shape[1] + real_y , 0]
        comp_img[0:comp_img.shape[0], 0:comp_img.shape[1], 1] = new_curr_img[area_size + real_x : area_size + comp_img.shape[0] + real_x , area_size + real_y : area_size + comp_img.shape[1] + real_y , 1]
        comp_img[0:comp_img.shape[0], 0:comp_img.shape[1], 2] = new_curr_img[area_size + real_x : area_size + comp_img.shape[0] + real_x , area_size + real_y : area_size + comp_img.shape[1] + real_y , 2]

    return comp_img


for idx in dataRange:
    frame_img = cv2.imread(data[idx])
    frame_img_BW = cv2.cvtColor(frame_img,cv2.COLOR_BGR2GRAY)
    # motion_x, motion_y = compute_block_matching(reference_img_BW, frame_img_BW, block_size, area_size)
    # motion_matrix = np.zeros([frame_img_BW.shape[0] / block_size, frame_img_BW.shape[1] / block_size, 2])
    # motion_matrix[:,:,0] = motion_x
    # motion_matrix[:,:,1] = motion_y
    motion_matrix = compute_block_matching2017(reference_img_BW, frame_img_BW)
    camera_motion_x = np.bincount((motion_matrix[:,:,0]+area_size).astype(np.int64).flatten()).argmax()-area_size
    camera_motion_y = np.bincount((motion_matrix[:,:,1]+area_size).astype(np.int64).flatten()).argmax()-area_size

    # Apply stabilization:
    frame_img_stab = apply_stabilization(camera_motion_x, -camera_motion_y, frame_img)
    reference_img_BW =  cv2.cvtColor(np.uint8(frame_img_stab),cv2.COLOR_BGR2GRAY)
    out.write(np.uint8(frame_img_stab))
    cv2.imwrite("stabilizated_images/stabilized"+str(idx)+".png",np.uint8(frame_img_stab))


