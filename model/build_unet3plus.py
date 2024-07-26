import tensorflow as tf

def conv_block(x, num_filters, act=True):
    x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding="same")(x)

    if act == True:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

    return x

def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    x = conv_block(x, num_filters)

    p = tf.keras.layers.MaxPool2D((2, 2))(x)
    return x, p

def build_unet3plus(input_shape, num_classes=1):
    """ Inputs """
    inputs = tf.keras.layers.Input(input_shape, name="input_layer")

    """ Pre-trained VGG16 Model """
    encoder = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    e1 = encoder.get_layer("input_layer").output
    e2 = encoder.get_layer("conv1_relu").output
    e3 = encoder.get_layer("conv2_block3_out").output
    e4 = encoder.get_layer("conv3_block4_out").output

    """ Bridge """
    e5 = encoder.get_layer("conv4_block6_out").output

    """ Decoder 4 """
    e1_d4 = tf.keras.layers.MaxPool2D((8, 8))(e1)
    e1_d4 = conv_block(e1_d4, 64)

    e2_d4 = tf.keras.layers.MaxPool2D((4, 4))(e2)
    e2_d4 = conv_block(e2_d4, 64)

    e3_d4 = tf.keras.layers.MaxPool2D((2, 2))(e3)
    e3_d4 = conv_block(e3_d4, 64)

    e4_d4 = conv_block(e4, 64)

    e5_d4 = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(e5)
    e5_d4 = conv_block(e5_d4, 64)

    d4 = tf.keras.layers.Concatenate()([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, 64*5)

    """ Decoder 3 """
    e1_d3 = tf.keras.layers.MaxPool2D((4, 4))(e1)
    e1_d3 = conv_block(e1_d3, 64)

    e2_d3 = tf.keras.layers.MaxPool2D((2, 2))(e2)
    e2_d3 = conv_block(e2_d3, 64)

    e3_d3 = conv_block(e3, 64)

    d4_d3 = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(d4)
    d4_d3 = conv_block(d4_d3, 64)

    e5_d3 = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(e5)
    e5_d3 = conv_block(e5_d3, 64)

    d3 = tf.keras.layers.Concatenate()([e1_d3, e2_d3, e3_d3, d4_d3, e5_d3])
    d3 = conv_block(d3, 64*5)

    """ Decoder 2 """
    e1_d2 = tf.keras.layers.MaxPool2D((2, 2))(e1)
    e1_d2 = conv_block(e1_d2, 64)

    e2_d2 = conv_block(e2, 64)

    d3_d2 = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(d3)
    d3_d2 = conv_block(d3_d2, 64)

    d4_d2 = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(d4)
    d4_d2 = conv_block(d4_d2, 64)

    e5_d2 = tf.keras.layers.UpSampling2D((8, 8), interpolation="bilinear")(e5)
    e5_d2 = conv_block(e5_d2, 64)

    d2 = tf.keras.layers.Concatenate()([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, 64*5)

    """ Decoder 1 """
    e1_d1 = conv_block(e1, 64)

    d2_d1 = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(d2)
    d2_d1 = conv_block(d2_d1, 64)

    d3_d1 = tf.keras.layers.UpSampling2D((4, 4), interpolation="bilinear")(d3)
    d3_d1 = conv_block(d3_d1, 64)

    d4_d1 = tf.keras.layers.UpSampling2D((8, 8), interpolation="bilinear")(d4)
    d4_d1 = conv_block(d4_d1, 64)

    e5_d1 = tf.keras.layers.UpSampling2D((16, 16), interpolation="bilinear")(e5)
    e5_d1 = conv_block(e5_d1, 64)

    d1 = tf.keras.layers.Concatenate()([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1])
    d1 = conv_block(d1, 64*5)

    """ Output """
    y1 = tf.keras.layers.Conv2D(num_classes, kernel_size=3, padding="same")(d1)
    y1 = tf.keras.layers.Activation("sigmoid")(y1)
    outputs = [y1]

    model = tf.keras.Model(inputs, outputs, name='UNET_3PLUS')
    return model