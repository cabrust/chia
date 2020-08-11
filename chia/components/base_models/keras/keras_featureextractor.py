import tensorflow as tf
from tensorflow.keras.applications import (
    inception_resnet_v2,
    mobilenet_v2,
    nasnet,
    resnet_v2,
)

from chia import components, helpers, instrumentation


class KerasFeatureExtractor(instrumentation.Observable):
    def __init__(
        self,
        side_length,
        use_pretrained_weights="ILSVRC2012",
        architecture="ResNet50V2",
        l2=5e-5,
        trainable=False,
    ):
        instrumentation.Observable.__init__(self)

        # Weights
        self.use_pretrained_weights = use_pretrained_weights

        # Architecture
        self.architecture = architecture
        if self.architecture == "ResNet50V2":
            self.feature_extractor = resnet_v2.ResNet50V2(
                include_top=False,
                input_tensor=None,
                input_shape=None,
                pooling="avg",
                weights="imagenet"
                if self.use_pretrained_weights == "ILSVRC2012"
                else None,
            )

        elif self.architecture == "InceptionResNetV2":
            self.feature_extractor = inception_resnet_v2.InceptionResNetV2(
                include_top=False,
                input_tensor=None,
                input_shape=None,
                pooling="avg",
                weights="imagenet"
                if self.use_pretrained_weights == "ILSVRC2012"
                else None,
            )

        elif self.architecture == "MobileNetV2":
            self.side_length = side_length
            self.feature_extractor = mobilenet_v2.MobileNetV2(
                include_top=False,
                input_tensor=None,
                input_shape=(self.side_length, self.side_length, 3),
                pooling="avg",
                weights="imagenet"
                if self.use_pretrained_weights == "ILSVRC2012"
                else None,
            )

        elif self.architecture == "NASNetMobile":
            self.side_length = side_length
            self.feature_extractor = nasnet.NASNetMobile(
                include_top=False,
                input_tensor=None,
                input_shape=(self.side_length, self.side_length, 3),
                pooling="avg",
                weights="imagenet"
                if self.use_pretrained_weights == "ILSVRC2012"
                else None,
            )

        else:
            raise ValueError(f'Unknown architecture "{self.architecture}"')

        # Load other pre-trained weights if necessary
        if (
            self.use_pretrained_weights is not None
            and self.use_pretrained_weights != "ILSVRC2012"
        ):
            self.log_info(
                f"Loading alternative pretrained weights {self.use_pretrained_weights}"
            )
            self.feature_extractor.load_weights(
                helpers.maybe_expand_path(self.use_pretrained_weights, self)
            )

        # Freezing of layers
        self.trainable = trainable
        if not self.trainable:
            for layer in self.feature_extractor.layers:
                layer.trainable = False

        # L2 Regularization
        self.l2_regularization = l2
        self._add_l2_regularizers()

    def _add_l2_regularizers(self):
        if self.l2_regularization == 0:
            return

        # Add regularizer:
        # see https://jricheimer.github.io/keras/2019/02/06/keras-hack-1/
        for layer in self.feature_extractor.layers:
            if (
                isinstance(layer, tf.keras.layers.Conv2D)
                and not isinstance(layer, tf.keras.layers.DepthwiseConv2D)
            ) or isinstance(layer, tf.keras.layers.Dense):
                layer.add_loss(
                    lambda layer=layer: tf.keras.regularizers.l2(
                        self.l2_regularization
                    )(layer.kernel)
                )
            elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                layer.add_loss(
                    lambda layer=layer: tf.keras.regularizers.l2(
                        self.l2_regularization
                    )(layer.depthwise_kernel)
                )
            if hasattr(layer, "bias_regularizer") and layer.use_bias:
                layer.add_loss(
                    lambda layer=layer: tf.keras.regularizers.l2(
                        self.l2_regularization
                    )(layer.bias)
                )

    def __call__(self, image_batch, training):
        return self.feature_extractor(image_batch, training)


class KerasFeatureExtractorFactory(components.Factory):
    name_to_class_mapping = KerasFeatureExtractor
    default_section = "keras_feature_extractor"
