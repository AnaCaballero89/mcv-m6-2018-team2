import csv
import pandas as pd
import os
import skvideo.io
from tqdm import tqdm

# constants
DATA_PATH = './'
TEST_VIDEO = os.path.join(DATA_PATH, 'myvideo.mp4')
CLEAN_DATA_PATH = './clean_data'
CLEAN_IMGS_TEST = os.path.join(CLEAN_DATA_PATH, 'myvideo_images')

test_frames = 8616


def dataset_constructor(video_loc, img_folder, tot_frames, dataset_type):
    meta_dict = {}

    tqdm.write('reading in video file...')
    cap = skvideo.io.vreader(video_loc)

    tqdm.write('constructing dataset...')
    for idx, frame in enumerate(tqdm(cap)):
        img_path = os.path.join(img_folder, str(idx) + '.jpg')
        frame_speed = float('NaN')
        meta_dict[idx] = [img_path, idx, frame_speed]
        skvideo.io.vwrite(img_path, frame)


    tqdm.write('writing meta to csv')
    with open(os.path.join(CLEAN_DATA_PATH, dataset_type + '_meta.csv'), 'wb') as f:
        w = csv.DictWriter(f, meta_dict.keys())
        w.writeheader()
        w.writerow(meta_dict)
    return "done dataset_constructor"

# test.mp4 data
dataset_constructor(TEST_VIDEO, CLEAN_IMGS_TEST, test_frames, 'myvideo.mp4')

#Check dataset construction
test_meta = pd.read_csv(os.path.join(CLEAN_DATA_PATH, 'myvideo.mp4_meta.csv'))
print test_meta.shape