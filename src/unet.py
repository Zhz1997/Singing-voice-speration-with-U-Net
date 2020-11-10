import tensorflow as tf
from tensorflow.keras.layers import concatenate, Dropout, BatchNormalization, Conv2D, Conv2DTranspose


def unet(features, labels, mode) :
    print("called")
    # Input Layer
    input_shape = (-1,512,128,1)
    input_layer = tf.reshape(features['mix'],input_shape)

    # # Convolutional Layer 1
    conv1 = BatchNormalization()(Conv2D(filters=16, kernel_size=[5,5],
                                                    strides=[2,2], padding="same", activation=tf.nn.leaky_relu,
                                                    input_shape=input_shape[1:])(input_layer))

    # # Convolutional Layer 2
    conv2 = BatchNormalization()(Conv2D(filters = 32, kernel_size = [5,5], strides = [2,2],
                                        padding="same", activation = tf.nn.leaky_relu)(conv1))

    # # Convolutional Layer 3
    conv3 = BatchNormalization()(Conv2D(filters = 64, kernel_size = [5,5], strides = [2,2],
                                        padding="same", activation = tf.nn.leaky_relu)(conv2))

    # # Convolutional Layer 4
    conv4 = BatchNormalization()(Conv2D(filters = 128, kernel_size = [5,5], strides = [2,2],
                                        padding="same", activation = tf.nn.leaky_relu)(conv3))

    # # Convolutional Layer 5
    conv5 = BatchNormalization()(Conv2D(filters = 256, kernel_size = [5,5], strides = [2,2],
                                        padding="same", activation = tf.nn.leaky_relu)(conv4))

    # # Convolutional Layer 6
    conv6 = BatchNormalization()(Conv2D(filters = 512, kernel_size = [5,5], strides = [2,2],
                                        padding="same", activation = tf.nn.leaky_relu)(conv5))

    # Deconvolutional Layer1 (dropout)
    deconv1 = BatchNormalization()(Conv2DTranspose(filters = 256, kernel_size = [5,5], strides = [2,2],
                                                 padding="same", activation = tf.nn.relu)(conv6))
    dropout1 = Dropout(rate = 0.5)(deconv1)

    # Deconvolutional Layer2 (dropout)
    deconv2 = BatchNormalization()(Conv2DTranspose(filters = 128, kernel_size = [5,5], strides = [2,2],
                                                 padding="same", activation = tf.nn.relu)(concatenate([dropout1,conv5],3)))
    dropout2 = Dropout(rate = 0.5)(deconv2)

    # Deconvolutional Layer3 (dropout)
    deconv3 = BatchNormalization()(Conv2DTranspose(filters = 64, kernel_size = [5,5], strides = [2,2],
                                                 padding="same", activation = tf.nn.relu)(concatenate([dropout2,conv4],3)))
    dropout3 = Dropout(rate = 0.5)(deconv3)

    # Deconvolutional Layer4
    deconv4 = BatchNormalization()(Conv2DTranspose(filters = 32, kernel_size = [5,5], strides = [2,2],
                                                 padding="same", activation = tf.nn.relu)(concatenate([dropout3,conv3],3)))

    # Deconvolutional Layer5
    deconv5 = BatchNormalization()(Conv2DTranspose(filters = 16, kernel_size = [5,5], strides = [2,2],
                                                 padding="same", activation = tf.nn.relu)(concatenate([deconv4,conv2],3)))
    # Deconvolutional Layer6
    deconv6 = Conv2DTranspose(filters = 1, kernel_size = [5,5], strides = [2,2],
                              padding="same", activation = tf.nn.relu)(concatenate([deconv5,conv1],3))

    predictions = {'outputs': deconv6
                  }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.compat.v1.losses.absolute_difference(labels,deconv6)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.compat.v1.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss)
