import tensorflow as tf

def ASPP(inputs):
    shape = inputs.shape

    y_pool = tf.keras.layers.AveragePooling2D(pool_size=(shape[1], shape[2]), name='average_pooling')(inputs)
    y_pool = tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = tf.keras.layers.BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = tf.keras.layers.Activation('relu', name=f'relu_1')(y_pool)
    y_pool = tf.keras.layers.UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(inputs)
    y_1 = tf.keras.layers.BatchNormalization()(y_1)
    y_1 = tf.keras.layers.Activation('relu')(y_1)

    y_6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(inputs)
    y_6 = tf.keras.layers.BatchNormalization()(y_6)
    y_6 = tf.keras.layers.Activation('relu')(y_6)

    y_12 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(inputs)
    y_12 = tf.keras.layers.BatchNormalization()(y_12)
    y_12 = tf.keras.layers.Activation('relu')(y_12)

    y_18 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(inputs)
    y_18 = tf.keras.layers.BatchNormalization()(y_18)
    y_18 = tf.keras.layers.Activation('relu')(y_18)

    y = tf.keras.layers.Concatenate()([y_pool, y_1, y_6, y_12, y_18])

    y = tf.keras.layers.Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    return y

def build_deeplabv3plus(shape):
    """ Inputs """
    inputs = tf.keras.layers.Input(shape)

    """ Pre-trained ResNet50 """
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    """ Pre-trained ResNet50 Output """
    image_features = base_model.get_layer('conv4_block6_out').output
    x_a = ASPP(image_features)
    x_a = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    """ Get low-level features """
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = tf.keras.layers.Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = tf.keras.layers.BatchNormalization()(x_b)
    x_b = tf.keras.layers.Activation('relu')(x_b)

    x = tf.keras.layers.Concatenate()([x_a, x_b])

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(x)

    x = tf.keras.layers.Conv2D(1, (1, 1), name='output_layer')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)

    """ Model """
    model = tf.keras.models.Model(inputs=inputs, outputs=x, name="DeepLabV3Plus")
    return model