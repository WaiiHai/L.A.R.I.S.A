from typing import Any
import tensorflow as tf

class DataLoader:
    PATH_ERROR_MESSAGE = 'The path to the folder doesnt exist.'

    def _load_image(self, path: str):
        tensor = tf.io.read_file(path)
        tensor = tf.io.decode_bmp(tensor, channels=3)
        return tensor
    
    def _normalize_image(self, tensor, size=(256, 256)):
        tensor = tf.image.resize(tensor, size)
        tensor = tf.image.convert_image_dtype(tensor, tf.float32)
        return tensor / 255.
    
    def _load_data(self, image, mask):
        image = self._load_image(image)
        image = self._normalize_image(image, (256, 256))

        mask = self._load_image(mask)
        mask = tf.image.rgb_to_grayscale(mask)
        mask = self._normalize_image(mask, (256, 256))

        return (image, mask)

    def __init__(self, path: str) -> None:
        import os
        if not os.path.exists(path): raise ValueError(self.PATH_ERROR_MESSAGE)

        import glob
        images = sorted(glob.glob(os.path.join(path, 'images', '*.bmp')))
        masks = sorted(glob.glob(os.path.join(path, 'masks', '*.bmp')))

        self.data = tf.data.Dataset.from_tensor_slices((images, masks))
        self.length = len(images)

    def __call__(self, batch_size: int, transforms=None):
        data = self.data.map(self._load_data, num_parallel_calls=tf.data.AUTOTUNE)
        from .augmentator import Augmentator
        if transforms is not None:
            augmentator = Augmentator(transforms)
            data = data.map(augmentator, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return (data, self.length)