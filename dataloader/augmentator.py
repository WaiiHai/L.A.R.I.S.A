import tensorflow as tf

class Augmentator:
    def __init__(self, transorms):
        self.transforms = transorms

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask

class RandomFlip:
    def __call__(self, image, mask):
        result_tensor = tf.concat((image, mask), axis=-1)
        result_tensor = tf.image.random_flip_left_right(result_tensor)
        return tf.split(result_tensor, (3, 1), axis=-1)

class RandomCrop:
    def __call__(self, image, mask):
        result = tf.concat((image, mask), axis=-1)

        random_crop = tf.random.uniform((), 0.3, 1)
        result = tf.image.central_crop(result, random_crop)
        return tf.split(tf.image.resize(result, (256, 256)), (3, 1), axis=-1)

class RandomRotation:
    def __call__(self, image, mask):
        result = tf.concat((image, mask), axis=-1)

        random_angle = tf.random.uniform((), 0, 0.8)
        import tensorflow_addons as tfa
        result = tfa.image.rotate(result, angles=random_angle, interpolation='NEAREST')
        return tf.split(tf.image.resize(result, (256, 256)), (3, 1), axis=-1)