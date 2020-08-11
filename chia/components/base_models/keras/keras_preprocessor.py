import numpy as np
import tensorflow as tf

from chia import components


class KerasPreprocessor:
    def __init__(
        self,
        augmentation,
        random_crop_to_size=None,
        channel_mean=(127.5, 127.5, 127.5),
        channel_stddev=(127.5, 127.5, 127.5),
    ):
        # Preprocessing
        self.random_crop_to_size = random_crop_to_size
        _channel_mean = channel_mean
        self.channel_mean_normalized = np.array(_channel_mean) / 255.0
        _channel_stddev = channel_stddev
        self.channel_stddev_normalized = np.array(_channel_stddev) / 255.0

        # Augmentation
        self.augmentation = augmentation

    def preprocess_image_batch(self, image_batch, is_training):
        image_batch = tf.cast(image_batch, dtype=tf.float32) / 255.0

        if is_training:
            # Processing_fn expects values in [0, 1]
            image_batch = self.augmentation.process(image_batch)

        # Map to correct range, e.g. [-1.0 , 1.0]
        image_batch = image_batch - self.channel_mean_normalized
        image_batch = image_batch / self.channel_stddev_normalized

        # Do cropping here instead of in augmentation because all augmentation is
        # disabled during testing...
        if self.random_crop_to_size is not None:
            if is_training:
                image_batch = tf.map_fn(self._random_crop_single_image, image_batch)
            else:
                image_batch = tf.image.crop_to_bounding_box(
                    image_batch,
                    (image_batch.shape[1] - self.random_crop_to_size[1]) // 2,
                    (image_batch.shape[2] - self.random_crop_to_size[0]) // 2,
                    self.random_crop_to_size[1],
                    self.random_crop_to_size[0],
                )

        return image_batch

    def _random_crop_single_image(self, image):
        return tf.image.random_crop(
            image, [self.random_crop_to_size[1], self.random_crop_to_size[0], 3]
        )


class KerasPreprocessorFactory(components.Factory):
    name_to_class_mapping = KerasPreprocessor
    default_section = "keras_preprocessor"
