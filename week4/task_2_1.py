import glob
from block_matching import *

mode = 'folder'
ID = "Traffic"
path = "datasets/traffic/input/"
path_gt = "datasets/traffic/groundtruth/"
alpha = 1.9
rho = 0.03

data = sorted(glob.glob(path + "*.jpg"))
data_gt =  sorted(glob.glob(path_gt + "*.png"))
minFrame = 950
maxFrame = 1050
dataRange = np.arange(minFrame,maxFrame)

reference_frame = 950
reference_img = cv2.imread(data[reference_frame])
reference_img_BW = cv2.cvtColor(reference_img,cv2.COLOR_BGR2GRAY)

block_size = 16
area_size = 32

# Define the codec and create VideoWriter object
video_path = "stabilizated_videos/"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path+"task_2_1_"+".avi", fourcc, 20, (reference_img_BW.shape[0], reference_img_BW.shape[1]),1)


def apply_stabilization(real_x, real_y, curr_img):
    x_size = curr_img.shape[0]
    y_size = curr_img.shape[1]

    new_curr_img = np.zeros([x_size + 2 * area_size, y_size + 2 * area_size, 3])
    auxImg = np.pad(curr_img[:, :, 0], ((area_size, area_size), (area_size, area_size)), 'symmetric')
    new_curr_img[:, :, 0] = auxImg
    auxImg = np.pad(curr_img[:, :, 1], ((area_size, area_size), (area_size, area_size)), 'symmetric')
    new_curr_img[:, :, 1] = auxImg
    auxImg = np.pad(curr_img[:, :, 2], ((area_size, area_size), (area_size, area_size)), 'symmetric')
    new_curr_img[:, :, 2] = auxImg
    comp_img = np.zeros([x_size, y_size, 3])

    comp_img[0:comp_img.shape[0], 0:comp_img.shape[1], 0] = new_curr_img[area_size + real_x : area_size + comp_img.shape[0] + real_x , area_size + real_y : area_size + comp_img.shape[1] + real_y , 0]
    comp_img[0:comp_img.shape[0], 0:comp_img.shape[1], 1] = new_curr_img[area_size + real_x : area_size + comp_img.shape[0] + real_x , area_size + real_y : area_size + comp_img.shape[1] + real_y , 1]
    comp_img[0:comp_img.shape[0], 0:comp_img.shape[1], 2] = new_curr_img[area_size + real_x : area_size + comp_img.shape[0] + real_x , area_size + real_y : area_size + comp_img.shape[1] + real_y , 2]

    return comp_img


for idx in dataRange:
    frame_img = cv2.imread(data[idx])
    frame_img_BW = cv2.cvtColor(frame_img,cv2.COLOR_BGR2GRAY)
    frame_gt = cv2.imread(data_gt[idx])

    motion_x, motion_y = compute_block_matching(reference_img_BW, frame_img_BW, block_size, area_size)
    motion_matrix = np.zeros([frame_img_BW.shape[0] / block_size, frame_img_BW.shape[1] / block_size, 2])
    motion_matrix[:,:,0] = motion_x
    motion_matrix[:,:,1] = motion_y
    #motion_matrix = compute_block_matching2017(reference_img_BW, frame_img_BW)
    camera_motion_x = np.bincount((motion_matrix[:,:,0]+area_size).astype(np.int64).flatten()).argmax()-area_size
    camera_motion_y = np.bincount((motion_matrix[:,:,1]+area_size).astype(np.int64).flatten()).argmax()-area_size

    # Apply stabilization:
    frame_img_stab = apply_stabilization(camera_motion_x, -camera_motion_y, frame_img)
    frame_gt_stab = apply_stabilization(camera_motion_x, -camera_motion_y, frame_gt)
    reference_img_BW =  cv2.cvtColor(np.uint8(frame_img_stab),cv2.COLOR_BGR2GRAY)
    out.write(np.uint8(frame_img_stab))
    cv2.imwrite("stabilizated_images/stabilized"+str(idx)+".jpg",np.uint8(frame_img_stab))
    cv2.imwrite("stabilizated_gt/stabilized_gt_"+str(idx)+".png",np.uint8(frame_gt_stab))


