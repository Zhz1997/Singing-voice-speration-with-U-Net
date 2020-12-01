from config import *
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, LeakyReLU, Conv2DTranspose as Deconv2D, Activation, Concatenate


def unet(inputs=Input((512, 128, 1))):
    conv1 = Conv2D(16, 5, strides=2, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)

    conv2 = Conv2D(32, 5, strides=2, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)

    conv3 = Conv2D(64, 5, strides=2, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)

    conv4 = Conv2D(128, 5, strides=2, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)

    conv5 = Conv2D(256, 5, strides=2, padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)

    conv6 = Conv2D(512, 5, strides=2, padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)

    deconv7 = Deconv2D(256, 5, strides=2, padding='same')(conv6)
    deconv7 = BatchNormalization()(deconv7)
    deconv7 = Dropout(0.5)(deconv7)
    deconv7 = Activation('relu')(deconv7)

    deconv8 = Concatenate(axis=3)([deconv7, conv5])
    deconv8 = Deconv2D(128, 5, strides=2, padding='same')(deconv8)
    deconv8 = BatchNormalization()(deconv8)
    deconv8 = Dropout(0.5)(deconv8)
    deconv8 = Activation('relu')(deconv8)

    deconv9 = Concatenate(axis=3)([deconv8, conv4])
    deconv9 = Deconv2D(64, 5, strides=2, padding='same')(deconv9)
    deconv9 = BatchNormalization()(deconv9)
    deconv9 = Dropout(0.5)(deconv9)
    deconv9 = Activation('relu')(deconv9)

    deconv10 = Concatenate(axis=3)([deconv9, conv3])
    deconv10 = Deconv2D(32, 5, strides=2, padding='same')(deconv10)
    deconv10 = BatchNormalization()(deconv10)
    deconv10 = Activation('relu')(deconv10)

    deconv11 = Concatenate(axis=3)([deconv10, conv2])
    deconv11 = Deconv2D(16, 5, strides=2, padding='same')(deconv11)
    deconv11 = BatchNormalization()(deconv11)
    deconv11 = Activation('relu')(deconv11)

    deconv12 = Concatenate(axis=3)([deconv11, conv1])
    deconv12 = Deconv2D(1, 5, strides=2, padding='same')(deconv12)
    deconv12 = Activation('relu')(deconv12)

    model = Model(inputs=inputs, outputs=deconv12)
    model.summary()
    return model

