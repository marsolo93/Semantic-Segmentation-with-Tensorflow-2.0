import tensorflow as tf
import tensorflow.keras.backend as K

def weighted_categorical_crossentropy(weights):
    def func(y_true, y_pred):
        Kweights = tf.keras.backend.constant(weights)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.keras.backend.categorical_crossentropy(y_true, y_pred) * tf.keras.backend.sum(y_true * Kweights, axis=-1))
    return func

def meanIOU():
    def func(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
        return K.mean((intersection + 1) / (union + 1), axis=0)
    return func

def dice_coeff():
    def func(y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return func