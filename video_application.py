import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from models import *
from tensorflow.python.framework.ops import disable_eager_execution
import time
from config import *
from utils import *
import os
import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

video = 'stuttgart_00'
path = 'C:\\Users\\Marcel\\FU-Berlin\\object_segmentation\\'
model_name = 'mobilenet_swiftnet'
weights_path = path + model_name + '.h5'
pic_path = 'C:\\Users\\Marcel\\FU-Berlin\\object_segmentation\\demoVideo\\' + video + '\\'
video_path = path + model_name + '_' + video + '.mp4'

model_config = 'SWIFTNET'

model = load_model_w_weights(model_config, weights_path)

project_dir = "C:\\Users\\Marcel\\FU-Berlin\\object_segmentation\\project_data\\"

means = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])

list_img = os.listdir(pic_path)
print(list_img)

width, height = 512, 256

final_sequence = []

for i, img in tqdm.tqdm(enumerate(list_img)):
    img_ = cv2.imread(pic_path + img, -1)[..., ::-1]

    start = time.time()
    img_show = cv2.resize(img_, (256, 256), interpolation=cv2.INTER_NEAREST)
    img_show = (img_show / 255.0 - means) / std
    img_show_reshaped = np.reshape(img_show, [1, 256, 256, 3])
    out = model.predict(img_show_reshaped)
    out_reshaped = np.reshape(out, [256, 256, 19])
    out_read = post_processing(out_reshaped, one_hot=True)
    out_read = cv2.resize(out_read, (width, height), interpolation=cv2.INTER_NEAREST)
    img_ = cv2.resize(img_, (width, height), interpolation=cv2.INTER_NEAREST)

    img_final = cv2.addWeighted(np.asarray(img_, np.int32), 0.7, np.asarray(out_read, np.int32), 0.3, 0)

    final_sequence.append(img_final)

    stop = time.time()

clip = ImageSequenceClip(final_sequence, fps=24)
clip.write_videofile(video_path)
