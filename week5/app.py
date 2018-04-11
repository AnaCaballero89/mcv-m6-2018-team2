import csv
import pandas as pd
import cPickle as pickle
import os
import skvideo.io
import cv2
from tqdm import tqdm

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

# constants
DATA_PATH = './'
TEST_VIDEO = os.path.join(DATA_PATH, 'myvideo.mp4')
CLEAN_DATA_PATH = './clean_data'
CLEAN_IMGS_TEST = os.path.join(CLEAN_DATA_PATH, 'myvideo_images')


def unpickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

writer = skvideo.io.FFmpegWriter("outputvideo.mp4")
tqdm.write('reading in video file...')
cap = skvideo.io.vreader(TEST_VIDEO)
tqdm.write('Generating video...')
predictions = unpickle('./predictions_myvideo.pkl')
speed_limit = 16 #in metres per second
speed = None
for idx, frame in enumerate(tqdm(cap)):
    #txt = Image.new('RGBA', (512, 512), (255, 255, 255, 0))
    try:
        speed = int(predictions[str(idx)])
        tqdm.write(str(speed))
    except(KeyError):
        pass
    """draw = ImageDraw.Draw(txt)
    if speed < speed_limit:
        draw.text((0, 0), "Speed: {} m/s".format(speed), fill=(0, 255, 0, 0))
    else:
        draw.text((0, 0), "Speed: {} m/s".format(speed), fill=(255, 0, 0, 0))
    out = Image.alpha_composite(Image.open(os.path.join(CLEAN_IMGS_TEST, str(idx)+'.jpg')).convert('RGBA'), txt)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    if speed < speed_limit:
        cv2.putText(frame, "Speed: {} m/s".format(speed), (40, 40), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Speed: {} m/s".format(speed), (40, 40), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    writer.writeFrame(frame)
writer.close()