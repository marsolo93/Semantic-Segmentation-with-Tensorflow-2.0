from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, Concatenate, Input, AveragePooling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.compat.v1.image import resize_bilinear
from classification_models import Classifiers
import tensorflow as tf
import os
import math
import config

project_dir = config.SAVE_PATH

class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, filter, kernel_size, stride, padding='valid', l2=False, dilation=1, bn=True, act=True, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filter = filter
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.l2 = l2
        self.dilation = dilation
        self.act = act
        self.bn = bn
        if l2:
            self.conv = Conv2D(self.filter, kernel_size=(self.kernel_size, self.kernel_size),
                               strides=(self.stride, self.stride), padding=self.padding, kernel_regularizer=regularizers.l2(self.l2), dilation_rate=self.dilation)
        else:
            self.conv = Conv2D(self.filter, kernel_size=(self.kernel_size, self.kernel_size),
                               strides=(self.stride, self.stride), padding=self.padding, dilation_rate=self.dilation)
        if bn:
            self.batch = BatchNormalization()

        self.relu = Activation('relu')

    def call(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batch(x)
        if self.act:
            x = self.relu(x)
        return x

    def get_config(self):
        config = super(ConvBlock, self).get_config()
        config.update({"filter": self.filter})
        config.update({"kernel_size": self.kernel_size})
        config.update({"stride": self.stride})
        config.update({"padding": self.padding})
        config.update({"l2": self.l2})
        config.update({"dilation": self.dilation})
        return config


class UpSamplingBlock(tf.keras.layers.Layer):

    def __init__(self, filter, l2=False, **kwargs):
        super(UpSamplingBlock, self).__init__(**kwargs)
        self.filter = filter
        self.l2 = l2
        self.upsample = tf.keras.layers.Conv2DTranspose(self.filter, kernel_size=(2, 2), strides=(2, 2))
        self.batch = BatchNormalization()
        self.conv_1 = ConvBlock(self.filter, kernel_size=3, stride=1, padding='same', l2=self.l2)
        self.conv_2 = ConvBlock(self.filter, kernel_size=3, stride=1, padding='same', l2=self.l2)
        self.conv_1x1 = Conv2D(self.filter, kernel_size=(1, 1), padding='same', kernel_regularizer=regularizers.l2(self.l2))

    def call(self, input):
        x, skip = input
        x = self.upsample(x)
        x = self.batch(x)
        x = Concatenate()([x, skip])
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

    def get_config(self):
        config = super(UpSamplingBlock, self).get_config()
        config.update({"filter": self.filter})
        config.update({"l2": self.l2})
        return config

class ResidualConvBlock(tf.keras.layers.Layer):

    def __init__(self, filter1, filter2, dilation, l2=False, **kwargs):
        super(ResidualConvBlock, self).__init__(**kwargs)
        self.filter1 = filter1
        self.filter2 = filter2
        self.dilation = dilation
        self.l2 = l2
        if l2:
            self.conv_block_1 = ConvBlock(filter1, 1, 1, padding='same', l2=l2, dilation=1)
            self.conv_block_2 = ConvBlock(filter1, 3, 1, padding='same', l2=l2, dilation=dilation)

            self.conv_skip = ConvBlock(filter2, 1, 1, padding='same', l2=l2, dilation=1)

            self.conv_3 = Conv2D(filter2, kernel_size=(1, 1),
                                   strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(l2), dilation_rate=1)
        else:
            self.conv_block_1 = ConvBlock(filter1, 1, 1, padding='same', dilation=1)
            self.conv_block_2 = ConvBlock(filter1, 3, 1, padding='same', dilation=dilation)

            self.conv_skip = ConvBlock(filter2, 1, 1, padding='same', dilation=1)

            self.conv_3 = Conv2D(filter2, kernel_size=(1, 1),
                                 strides=(1, 1), padding='same', dilation_rate=1)
        self.bn = BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.relu = Activation('relu')

    def call(self, x):
        skip = x
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        skip = self.conv_skip(skip)

        x = self.conv_3(x)
        x = self.bn(x)
        x = self.add([x, skip])
        x = self.relu(x)
        return x

    def get_config(self):
        config = super(ResidualConvBlock, self).get_config()
        config.update({"filter1": self.filter1})
        config.update({"filter2": self.filter2})
        config.update({"dilation": self.dilation})
        config.update({"l2": self.l2})
        return config



def resnet18(input_shape = [256, 512, 3], include_top=False, weights='imagenet'):
    print('Load ResNet18...')
    ResNet18, _ = Classifiers.get('resnet18')
    model = ResNet18(input_shape=input_shape,
                     include_top=include_top,
                     weights=weights)
    print('ResNet18 Summary:')
    model.summary()
    return model

def resnet50(input_shape = [256, 512, 3], include_top=False, weights='imagenet'):
    print('Load ResNet50...')
    model = ResNet50(input_shape=input_shape,
                     include_top=include_top,
                     weights=weights)
    print('ResNet50 Summary:')
    model.summary()
    return model

def mobilenet(input_shape = [256, 512, 3], include_top=False, weights='imagenet'):
    print('Load MobileNet...')
    model = MobileNetV2(input_shape=input_shape,
                        include_top=include_top,
                        weights=weights)
    print('MobileNet Summary:')
    model.summary()
    return model

def UpSampling2DBilinear(shape=[256, 512], **kwargs):
    def layer(x):
        return resize_bilinear(x, shape, align_corners=True)
    return Lambda(layer, **kwargs)

class PoolingPath(tf.keras.layers.Layer):
    def __init__(self, grid_size, height, width, l2=False, **kwargs):
        super(PoolingPath, self).__init__(**kwargs)
        self.grid_size = grid_size
        self.l2 = l2
        self.height = height // 32
        self.width = width // 32
        self.pool_h = self.height // self.grid_size
        self.pool_w = self.width // self.grid_size
        self.average_pooling = AveragePooling2D(pool_size=(self.pool_h, self.pool_w),
                                                strides=(self.pool_h, self.pool_w),
                                                padding='valid')
        self.batch = BatchNormalization()
        self.relu = Activation('relu')
        if self.l2:
            self.conv = Conv2D(64, kernel_size=1, padding='same', kernel_regularizer=regularizers.l2(self.l2))
        else:
            self.conv = Conv2D(64, kernel_size=1, padding='same')

    def call(self, x):
        x = self.average_pooling(x)
        x = self.batch(x)
        x = self.relu(x)
        x = self.conv(x)
        x = resize_bilinear(x, [self.height, self.width], align_corners=True)
        return x

    def get_config(self):
        config = super(PoolingPath, self).get_config()
        config.update({"grid_size": self.grid_size})
        config.update({"l2": self.l2})
        config.update({"height": self.height})
        config.update({"width": self.width})
        return config


class SpatialPyramidPooling(tf.keras.layers.Layer):
    def __init__(self, height, width, grids=(8, 4, 2), l2=False, **kwargs):
        super(SpatialPyramidPooling, self).__init__(**kwargs)
        self.l2 = l2
        self.grids = grids
        self.batch_prep = BatchNormalization()
        self.relu_prep = Activation('relu')
        self.height = height
        self.width = width
        if self.l2:
            self.conv_prep = Conv2D(128, kernel_size=1, padding='same', kernel_regularizer=regularizers.l2(self.l2))
            self.conv = Conv2D(128, kernel_size=1, padding='same', kernel_regularizer=regularizers.l2(self.l2))
        else:
            self.conv_prep = Conv2D(128, kernel_size=1, padding='same')
            self.conv = Conv2D(128, kernel_size=1, padding='same')
        self.path_grid_8 = PoolingPath(grids[0], self.height, self.width, self.l2)
        self.path_grid_4 = PoolingPath(grids[1], self.height, self.width, self.l2)
        self.path_grid_2 = PoolingPath(grids[2], self.height, self.width, self.l2)
        self.concat = Concatenate()
        self.batch = BatchNormalization()
        self.relu = Activation('relu')

    def call(self, x):
        x = self.batch_prep(x)
        x = self.relu_prep(x)
        x = self.conv_prep(x)
        x_8 = self.path_grid_8(x)
        x_4 = self.path_grid_4(x)
        x_2 = self.path_grid_2(x)
        x = self.concat([x_8, x_4, x_2])
        x = self.batch(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

    def get_config(self):
        config = super(SpatialPyramidPooling, self).get_config()
        config.update({"height": self.height})
        config.update({"width": self.width})
        config.update({"grids": self.grids})
        config.update({"l2": self.l2})
        return config

class AtrousPyramidPooling(tf.keras.layers.Layer):
    def __init__(self, filter, l2=False, dilation_rates=(6, 12, 18), **kwargs):
        super(AtrousPyramidPooling, self).__init__(**kwargs)
        self.filter = filter
        self.l2 = l2
        self.dilation_rates = dilation_rates
        self.conv1 = ConvBlock(self.filter, 1, 1, padding='valid', l2=self.l2, dilation=1, act=True)
        self.conv_atr_1 = ConvBlock(self.filter, 3, 1, padding='same', l2=self.l2, dilation=self.dilation_rates[0], act=True)
        self.conv_atr_2 = ConvBlock(self.filter, 3, 1, padding='same', l2=self.l2, dilation=self.dilation_rates[1], act=True)
        self.conv_atr_3 = ConvBlock(self.filter, 3, 1, padding='same', l2=self.l2, dilation=self.dilation_rates[2], act=True)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape = tf.keras.layers.Reshape([1, 1, self.filter])
        self.conv_pooling_post = ConvBlock(self.filter, 1, 1, padding='same', l2=self.l2, dilation=1, act=True)

        self.conv_post = ConvBlock(self.filter, 1, 1, padding='same', l2=self.l2, dilation=1, act=True)
        self.drop = Dropout(0.1)

    def call(self, x):
        x_1 = self.conv1(x)
        x_at_1 = self.conv_atr_1(x)
        x_at_2 = self.conv_atr_2(x)
        x_at_3 = self.conv_atr_3(x)
        x_pool = self.pool(x)
        x_pool = self.reshape(x_pool)
        x_pool = resize_bilinear(x_pool, [tf.shape(x_1)[1], tf.shape(x_1)[2]], align_corners=True)
        x_pool = self.conv_pooling_post(x_pool)
        x_concat = Concatenate()([x_1, x_at_1, x_at_2, x_at_3, x_pool])
        x = self.conv_post(x_concat)
        x = resize_bilinear(x, [tf.shape(x_1)[1] * 2, tf.shape(x_1)[2] * 2], align_corners=True)
        x = self.drop(x)
        return x

    def get_config(self):
        config = super(AtrousPyramidPooling, self).get_config()
        config.update({"filter": self.filter})
        config.update({"l2": self.l2})
        config.update({"dilation_rates": self.dilation_rates})
        return config


def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(project_dir, 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

class UpSampleSwift(tf.keras.layers.Layer):
    def __init__(self, filter, l2=False, **kwargs):
        super(UpSampleSwift, self).__init__(**kwargs)
        self.l2 = l2
        self.filter = filter
        self.batch_skip = BatchNormalization()
        self.batch_add = BatchNormalization()
        self.relu_skip = Activation('relu')
        self.relu_add = Activation('relu')
        if self.l2:
            self.conv_skip = Conv2D(self.filter, kernel_size=1, padding='same',
                                    kernel_regularizer=regularizers.l2(self.l2))
            self.conv_add = Conv2D(self.filter, kernel_size=3, padding='same',
                                    kernel_regularizer=regularizers.l2(self.l2))
        else:
            self.conv_skip = Conv2D(self.filter, kernel_size=1, padding='same')
            self.conv_add = Conv2D(self.filter, kernel_size=3, padding='same')

    def call(self, input):
        x, skip = input
        # Skip preparation
        skip = self.batch_skip(skip)
        skip = self.relu_skip(skip)
        skip = self.conv_skip(skip)

        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        x = resize_bilinear(x, [height * 2, width * 2], align_corners=True)
        x = x + skip
        x = self.batch_add(x)
        x = self.relu_add(x)
        x = self.conv_add(x)
        return x

    def get_config(self):
        config = super(UpSampleSwift, self).get_config()
        config.update({"filter": self.filter})
        config.update({"l2": self.l2})
        return config

def build_swiftnet_model(height, width, channels, num_classes, l2_scale=None, network='mobilenet'):

    if network == 'resnet50':
        encoder_model = resnet50(input_shape=[height, width, channels])

        encoder_model = add_regularization(encoder_model, regularizer=tf.keras.regularizers.l2(l2_scale / 4))


        for layer in encoder_model.layers:
            layer._name = 'encoder_' + layer._name

        skip_8_16 = encoder_model.get_layer('encoder_conv5_block3_out').output
        skip_16_32 = encoder_model.get_layer('encoder_conv4_block6_out').output
        skip_32_64 = encoder_model.get_layer('encoder_conv3_block4_out').output
        skip_64_128 = encoder_model.get_layer('encoder_conv2_block3_out').output
        skip_128_256 = encoder_model.get_layer('encoder_conv1_relu').output

    elif network == 'resnet18':
        encoder_model = resnet18(input_shape=[height, width, channels])

        for layer in encoder_model.layers:
            layer._name = 'encoder_' + layer._name

        skip_8_16 = encoder_model.get_layer('encoder_add_8').output
        skip_16_32 = encoder_model.get_layer('encoder_add_6').output
        skip_32_64 = encoder_model.get_layer('encoder_add_4').output
        skip_64_128 = encoder_model.get_layer('encoder_add_2').output
        skip_128_256 = encoder_model.get_layer('encoder_conv0').output

        encoder_model = add_regularization(encoder_model, regularizer=tf.keras.regularizers.l2(l2_scale / 4))

    else:
        encoder_model = mobilenet(input_shape=[height, width, channels])

        encoder_model = add_regularization(encoder_model, regularizer=tf.keras.regularizers.l2(l2_scale / 4))

        for layer in encoder_model.layers:
            layer._name = 'encoder_' + layer._name

        skip_8_16 = encoder_model.get_layer('encoder_block_15_add').output
        skip_16_32 = encoder_model.get_layer('encoder_block_12_add').output
        skip_32_64 = encoder_model.get_layer('encoder_block_5_add').output
        skip_64_128 = encoder_model.get_layer('encoder_block_2_add').output
        skip_128_256 = encoder_model.get_layer('encoder_Conv1').output

    x = SpatialPyramidPooling(config.HEIGHT, config.WIDTH, grids=(8, 4, 2), l2=l2_scale)(skip_8_16)
    x = UpSampleSwift(128, l2=l2_scale)([x, skip_16_32])
    x = UpSampleSwift(128, l2=l2_scale)([x, skip_32_64])
    x = UpSampleSwift(128, l2=l2_scale)([x, skip_64_128])
    x = UpSampleSwift(128, l2=l2_scale)([x, skip_128_256])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_classes, kernel_size=3, padding='same')(x)
    x = UpSampling2DBilinear(shape=[height, width])(x)
    final = Activation('softmax')(x)

    model = Model(inputs=[encoder_model.input],
                  outputs=[final])
    return model

def build_deeplab3plus_model(height, width, channels, num_classes, l2_scale=None, network='mobilenet'):

    if network == 'resnet50':
        encoder_model = resnet50(input_shape=[height, width, channels])

        encoder_model = add_regularization(encoder_model, regularizer=tf.keras.regularizers.l2(l2_scale / 4))


        for layer in encoder_model.layers:
            layer._name = 'encoder_' + layer._name

        skip_32_64 = encoder_model.get_layer('encoder_conv3_block4_out').output
        skip_64_128 = encoder_model.get_layer('encoder_conv2_block3_out').output

    elif network == 'resnet18':
        encoder_model = resnet18(input_shape=[height, width, channels])

        for layer in encoder_model.layers:
            layer._name = 'encoder_' + layer._name

        skip_32_64 = encoder_model.get_layer('encoder_add_4').output
        skip_64_128 = encoder_model.get_layer('encoder_add_2').output

        encoder_model = add_regularization(encoder_model, regularizer=tf.keras.regularizers.l2(l2_scale / 4))

    else:
        encoder_model = mobilenet(input_shape=[height, width, channels])

        encoder_model = add_regularization(encoder_model, regularizer=tf.keras.regularizers.l2(l2_scale / 4))

        for layer in encoder_model.layers:
            layer._name = 'encoder_' + layer._name

        skip_32_64 = encoder_model.get_layer('encoder_block_5_add').output
        skip_64_128 = encoder_model.get_layer('encoder_block_2_add').output

    x = ResidualConvBlock(512, 1024, 2, l2=l2_scale)(skip_32_64)
    x = ResidualConvBlock(1024, 256, 4, l2=l2_scale)(x)

    x = AtrousPyramidPooling(256, l2=l2_scale, dilation_rates=(6, 12, 18))(x)

    x_low = ConvBlock(48, 1, 1, padding='same', l2=l2_scale)(skip_64_128)
    #
    x = Concatenate()([x_low, x])

    x = ConvBlock(304, 3, 1, padding='same', l2=l2_scale)(x)
    x = ConvBlock(256, 3, 1, padding='same', l2=l2_scale)(x)
    x = Conv2D(num_classes, kernel_size=1, padding='same')(x)
    x = UpSampling2DBilinear(shape=[config.HEIGHT, config.WIDTH])(x)
    final = Activation('softmax')(x)

    model = Model(inputs=[encoder_model.input],
                  outputs=[final])
    return model

def build_unet_model(height, width, channels, num_classes, l2_scale=None, network='mobilenet'):

    if network == 'resnet50':
        encoder_model = resnet50(input_shape=[height, width, channels])

        encoder_model = add_regularization(encoder_model, regularizer=tf.keras.regularizers.l2(l2_scale / 4))


        for layer in encoder_model.layers:
            layer._name = 'encoder_' + layer._name

        skip_8_16 = encoder_model.get_layer('encoder_conv5_block3_out').output
        skip_16_32 = encoder_model.get_layer('encoder_conv4_block6_out').output
        skip_32_64 = encoder_model.get_layer('encoder_conv3_block4_out').output
        skip_64_128 = encoder_model.get_layer('encoder_conv2_block3_out').output
        skip_128_256 = encoder_model.get_layer('encoder_conv1_relu').output

    elif network == 'resnet18':
        encoder_model = resnet18(input_shape=[height, width, channels])

        for layer in encoder_model.layers:
            layer._name = 'encoder_' + layer._name

        skip_8_16 = encoder_model.get_layer('encoder_add_8').output
        skip_16_32 = encoder_model.get_layer('encoder_add_6').output
        skip_32_64 = encoder_model.get_layer('encoder_add_4').output
        skip_64_128 = encoder_model.get_layer('encoder_add_2').output
        skip_128_256 = encoder_model.get_layer('encoder_conv0').output

        encoder_model = add_regularization(encoder_model, regularizer=tf.keras.regularizers.l2(l2_scale / 4))

    else:
        encoder_model = mobilenet(input_shape=[height, width, channels])

        encoder_model = add_regularization(encoder_model, regularizer=tf.keras.regularizers.l2(l2_scale / 4))

        for layer in encoder_model.layers:
            layer._name = 'encoder_' + layer._name

        skip_8_16 = encoder_model.get_layer('encoder_block_15_add').output
        skip_16_32 = encoder_model.get_layer('encoder_block_12_add').output
        skip_32_64 = encoder_model.get_layer('encoder_block_5_add').output
        skip_64_128 = encoder_model.get_layer('encoder_block_2_add').output
        skip_128_256 = encoder_model.get_layer('encoder_Conv1').output

    x = UpSamplingBlock(1024, l2=l2_scale)([skip_8_16, skip_16_32])
    x = UpSamplingBlock(512, l2=l2_scale)([x, skip_32_64])
    x = UpSamplingBlock(256, l2=l2_scale)([x, skip_64_128])
    x = UpSamplingBlock(64, l2=l2_scale)([x, skip_128_256])
    x = Conv2D(num_classes, kernel_size=1, padding='same')(x)
    x = UpSampling2DBilinear(shape=[config.HEIGHT, config.WIDTH])(x)
    final = Activation('softmax')(x)

    model = Model(inputs=[encoder_model.input],
                  outputs=[final])
    return model





