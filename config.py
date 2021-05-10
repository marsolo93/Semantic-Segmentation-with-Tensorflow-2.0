import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 4
NUM_BATCH = 2975 // BATCH_SIZE
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
LR = 1e-4
EPOCHS = 1
WARMUP_STEPS = 100
DECAY_STEPS = int(WARMUP_STEPS * NUM_BATCH)
LEARNING_RATE_END = 1e-5
NETWORK_BACKBONE = 'mobilenet'
SEGMENTOR = 'DEEPLABV3'

##  Setup data input pipeline

TRAINING_PATH = 'C:\\Users\\Marcel\\FU-Berlin\\object_segmentation\\project_data\\train\\training_df.csv'
PROBS_PATH = 'C:\\Users\\Marcel\\FU-Berlin\\object_segmentation\\project_data\\train\\train_prob_dict.pickle'
VAL_PATH = 'C:\\Users\\Marcel\\FU-Berlin\\object_segmentation\\project_data\\val\\val_df.csv'
#
# TRAINING_PATH = '/scratch/gauglitz/dnn/segmentation/semantic/project_data/train/training_df_curta.csv'
# PROBS_PATH = '/scratch/gauglitz/dnn/segmentation/semantic/project_data/train/train_prob_dict.pickle'
# VAL_PATH = '/scratch/gauglitz/dnn/segmentation/semantic/project_data/val/val_df_curta.csv'

WEIGHTS = np.array(
    [0.8373, 0.918, 0.866, 1.0345,
     1.0166, 0.9969, 0.9754, 1.0489,
     0.8786, 1.0023, 0.9539, 0.9843,
     1.1116, 0.9037, 1.0865, 1.0955,
     1.0865, 1.1529, 1.0507], dtype=np.float32)

CHANNEL_MEAN = np.load('C:\\Users\\Marcel\\FU-Berlin\\object_segmentation\\project_data\\train\\mean_colors.npy')

SAVE_NAME = 'mobile_segmentor'
SAVE_PATH = 'C:\\Users\\Marcel\\FU-Berlin\\object_segmentation\\'
#
# CHANNEL_MEAN = np.load('/scratch/gauglitz/dnn/segmentation/semantic/project_data/train/mean_colors.npy')
#
# SAVE_NAME = 'resnet_deeplabV3_segmentor'
# SAVE_PATH = '/scratch/gauglitz/dnn/segmentation/semantic/'

MEANS = np.asarray([0.485, 0.456, 0.406])#
STD = np.asarray([0.229, 0.224, 0.225])
