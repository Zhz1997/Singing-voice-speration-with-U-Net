import tensorflow as tf
from tensorflow.keras.layers import concatenate


def unet(features, labels, mode) :
    print("called")
    # Input Layer
    input_shape = (-1,512,128,1)
    input_layer = tf.reshape(features['mix'],input_shape)

    # # Convolutional Layer 1
    conv1 = tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv2D(filters=16, kernel_size=[5,5],
                                                    strides=[2,2], padding="same", activation=tf.nn.leaky_relu,
                                                    input_shape=input_shape[1:])(input_layer))

    # # Convolutional Layer 2
    conv2 = tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv2D(filters = 32, kernel_size = [5,5],
                                                    strides = [2,2], padding="same", activation = tf.nn.leaky_relu)(conv1))

    # # Convolutional Layer 3
    conv3 = tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv2D(filters = 64, kernel_size = [5,5],
                                                    strides = [2,2], padding="same", activation = tf.nn.leaky_relu)(conv2))

    # # Convolutional Layer 4
    conv4 = tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv2D(filters = 128, kernel_size = [5,5],
                                                    strides = [2,2], padding="same", activation = tf.nn.leaky_relu)(conv3))

    # # Convolutional Layer 5
    conv5 = tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv2D(filters = 256, kernel_size = [5,5],
                                                    strides = [2,2], padding="same", activation = tf.nn.leaky_relu)(conv4))

    # # Convolutional Layer 6
    conv6 = tf.keras.layers.BatchNormalization()(tf.keras.layers.Conv2D(filters = 512, kernel_size = [5,5],
                                                    strides = [2,2], padding="same", activation = tf.nn.leaky_relu)(conv5))

    # # Deconvolutional Layer1 (dropout)
    # deconv1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(inputs = conv6, filters = 256, kernel_size = [5,5],
    #                                                              strides = [2,2], padding="same", activation = tf.nn.relu))
    # dropout1 = tf.layers.dropout(inputs = deconv1, rate = 0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # # Deconvolutional Layer2 (dropout)
    # deconv2 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(inputs = concatenate([dropout1,conv5],3), filters = 128, kernel_size = [5,5],
    #                                                              strides = [2,2], padding="same", activation = tf.nn.relu))
    # dropout2 = tf.layers.dropout(inputs = deconv2, rate = 0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # # Deconvolutional Layer3 (dropout)
    # deconv3 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(inputs = concatenate([dropout2,conv4],3), filters = 64, kernel_size = [5,5],
    #                                                              strides = [2,2], padding="same", activation = tf.nn.relu))
    # dropout3 = tf.layers.dropout(inputs = deconv3, rate = 0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # # Deconvolutional Layer4
    # deconv4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(inputs = concatenate([dropout3,conv3],3), filters = 32, kernel_size = [5,5],
    #                                                              strides = [2,2], padding="same", activation = tf.nn.relu))

    # # Deconvolutional Layer5
    # deconv5 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(inputs = concatenate([deconv4,conv2],3), filters = 16, kernel_size = [5,5],
    #                                                              strides = [2,2], padding="same", activation = tf.nn.relu))
    # # Deconvolutional Layer6
    # deconv6 = tf.layers.conv2d_transpose(inputs = concatenate([deconv5,conv1],3), filters = 1, kernel_size = [5,5],
    #                                                                strides = [2,2], padding="same", activation = tf.nn.relu)

    # predictions = {'outputs': deconv6
    #               }

    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss = tf.losses.absolute_difference(labels,deconv6)

    # if mode == tf.estimator.ModeKeys.TRAIN:
    #     optimizer = tf.train.AdamOptimizer(1e-4)
    #     train_op = optimizer.minimize(
    #         loss=loss,
    #         global_step=tf.train.get_global_step())
    #     return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # return tf.estimator.EstimatorSpec(
    #     mode=mode, loss=loss)
