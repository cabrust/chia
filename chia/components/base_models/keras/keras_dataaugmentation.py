import math

import tensorflow as tf
import tensorflow_addons as tfa

from chia import components


class KerasDataAugmentation:
    def __init__(
        self,
        do_random_flip_horizontal=True,
        do_random_flip_vertical=False,
        do_random_rotate=False,
        do_random_crop=False,
        random_crop_factor=0.2,
        do_random_brightness_and_contrast=False,
        random_brightness_factor=0.05,
        random_contrast_factors=(0.7, 1.3),
        do_random_hue_and_saturation=False,
        random_hue_factor=0.08,
        random_saturation_factors=(0.6, 1.6),
        do_random_scale=False,
        random_scale_factors=(0.5, 2.0),
    ):

        self.do_random_flip_horizontal = do_random_flip_horizontal
        self.do_random_flip_vertical = do_random_flip_vertical

        self.do_random_rotate = do_random_rotate

        self.do_random_crop = do_random_crop
        self.random_crop_factor = random_crop_factor

        self.do_random_brightness_and_contrast = do_random_brightness_and_contrast
        self.random_brightness_factor = random_brightness_factor
        self.random_contrast_factors = random_contrast_factors

        self.do_random_hue_and_saturation = do_random_hue_and_saturation
        self.random_hue_factor = random_hue_factor
        self.random_saturation_factors = random_saturation_factors

        self.do_random_scale = do_random_scale
        self.random_scale_factors = random_scale_factors

    @tf.function
    def process(self, sample_batch):
        sample_batch = tf.map_fn(self._process_sample, sample_batch)
        return sample_batch

    @tf.function
    def _process_sample(self, sample):
        if self.do_random_flip_horizontal:
            sample = tf.image.random_flip_left_right(sample)
        if self.do_random_flip_vertical:
            sample = tf.image.random_flip_up_down(sample)

        if self.do_random_rotate:
            sample = tfa.image.rotate(
                sample,
                angles=tf.random.uniform(shape=[], minval=0, maxval=2.0 * math.pi),
                interpolation="BILINEAR",
            )

        if self.do_random_crop or self.do_random_scale:
            sample = self._inner_random_crop_or_scale(sample)

        if self.do_random_brightness_and_contrast:
            sample = tf.image.random_brightness(sample, self.random_brightness_factor)
            sample = tf.image.random_contrast(sample, *self.random_contrast_factors)

        if self.do_random_hue_and_saturation:
            sample = tf.image.random_hue(sample, self.random_hue_factor)
            sample = tf.image.random_saturation(sample, *self.random_saturation_factors)

        return sample

    @tf.function
    def _inner_random_crop_or_scale(self, x: tf.Tensor) -> tf.Tensor:
        if self.do_random_scale:
            scale = tf.random.uniform(
                shape=[],
                minval=self.random_scale_factors[0],
                maxval=self.random_scale_factors[1],
            )
        else:
            scale = 1.0

        width = 1.0 / scale
        height = 1.0 / scale

        if self.do_random_crop:
            left_min = -self.random_crop_factor
            left_max = 1.0 + self.random_crop_factor - width
            top_min = -self.random_crop_factor
            top_max = 1.0 + self.random_crop_factor - height
        else:
            left_min = 0.0
            left_max = 1.0 - width
            top_min = 0.0
            top_max = 1.0 - height

        left = tf.random.uniform(shape=[], minval=left_min, maxval=left_max)
        top = tf.random.uniform(shape=[], minval=top_min, maxval=top_max)
        boxes = [[top, left, top + height, left + width]]

        crop_shape = (x.shape[0], x.shape[1])
        # Create different crops for an image
        crops = tf.image.crop_and_resize(
            [x], boxes=boxes, box_indices=[0], crop_size=crop_shape
        )
        # Return a random crop
        return crops[0]


class KerasDataAugmentationFactory(components.Factory):
    name_to_class_mapping = KerasDataAugmentation
