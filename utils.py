import numpy as np
import tensorflow as tf
from swiftnet import *

def load_model_w_weights(detector_config, weights_path):
    if detector_config == 'SWIFTNET':
        loaded_model = tf.keras.models.load_model(weights_path, custom_objects={'ConvBlock':ConvBlock,
                                                                               'PoolingPath':PoolingPath,
                                                                               'SpatialPyramidPooling':SpatialPyramidPooling,
                                                                               'UpSampleSwift':UpSampleSwift})

    elif detector_config == 'DEEPLABV3':
        loaded_model = tf.keras.models.load_model(weights_path, custom_objects={'ConvBlock':ConvBlock,
                                                                               'ResidualConvBlock':ResidualConvBlock,
                                                                               'AtrousPyramidPooling':AtrousPyramidPooling})

    elif detector_config == 'UNET':
        loaded_model = tf.keras.models.load_model(weights_path, custom_objects={'UpSamplingBlock':UpSamplingBlock})

    print("Loaded model from disk")
    return loaded_model

def post_processing(preds, one_hot=True):
    if one_hot:
        preds = np.argmax(preds, axis=-1)
        print(preds)
    print(preds.shape)
    img = np.zeros([preds.shape[0], preds.shape[1], 3])
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            tmp = preds[i, j]
            if tmp == 0:
                img[i, j, :] = [153, 153, 1]
            elif tmp == 1:
                img[i, j, :] = [124, 1, 254]
            elif tmp == 2:
                img[i, j, :] = [254, 204, 204]
            elif tmp == 3:
                img[i, j, :] = [254, 204, 204]
            elif tmp == 4:
                img[i, j, :] = [254, 204, 204]
            elif tmp == 5:
                img[i, j, :] = [254, 1, 127]
            elif tmp == 6:
                img[i, j, :] = [254, 1, 127]
            elif tmp == 7:
                img[i, j, :] = [254, 1, 127]
            elif tmp == 8:
                img[i, j, :] = [1, 254, 1]
            elif tmp == 9:
                img[i, j, :] = [1, 254, 1]
            elif tmp == 10:
                img[i, j, :] = [1, 204, 204]
            elif tmp == 11:
                img[i, j, :] = [254, 1, 1]
            elif tmp == 12:
                img[i, j, :] = [254, 1, 1]
            elif tmp == 18:
                img[i, j, :] = [1, 200, 230]
            elif tmp == 19:
                img[i, j, :] = [1, 1, 1]
            else:
                img[i, j, :] = [1, 1, 254]
    return img