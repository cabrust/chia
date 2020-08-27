import tensorflow as tf

from chia import components
from chia.components.base_models.keras import (
    keras_basemodel,
    keras_dataaugmentation,
    keras_featureextractor,
    keras_learningrateschedule,
    keras_preprocessor,
    keras_trainer,
)


def _get_input_img_np(sample):
    return sample.get_resource("input_img_np")


class KerasOptimizerFactory(components.Factory):
    name_to_class_mapping = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
    }
    default_section = "keras_optimizer"
    i_know_that_var_args_are_not_supported = True


class KerasBaseModelContainer:
    def __init__(self, config, classifier, observers=()):
        self.learning_rate_schedule = (
            keras_learningrateschedule.KerasLearningRateScheduleFactory.create(
                config["learning_rate_schedule"], observers=observers
            )
        )
        self.optimizer = KerasOptimizerFactory.create(
            config["optimizer"], observers=observers
        )
        self.augmentation = keras_dataaugmentation.KerasDataAugmentationFactory.create(
            config["augmentation"] if "augmentation" in config.keys() else {},
            observers=observers,
        )
        self.preprocessor = keras_preprocessor.KerasPreprocessorFactory.create(
            config["preprocessor"] if "preprocessor" in config.keys() else {},
            observers=observers,
            augmentation=self.augmentation,
        )

        self.feature_extractor = (
            keras_featureextractor.KerasFeatureExtractorFactory.create(
                config["feature_extractor"],
                observers=observers,
                preprocessor=self.preprocessor,
            )
        )

        self.trainer = keras_trainer.KerasTrainerFactory.create(
            config["trainer"],
            observers=observers,
            feature_extractor=self.feature_extractor,
            preprocessor=self.preprocessor,
            classifier=classifier,
            learning_rate_schedule=self.learning_rate_schedule,
            optimizer=self.optimizer,
        )

        self.base_model = keras_basemodel.KerasBaseModelFactory.create(
            dict(),  # This is because a field "trainer" is present in config -> conflict
            observers=observers,
            batch_size=config["trainer"]["batch_size"],
            classifier=classifier,
            feature_extractor=self.feature_extractor,
            trainer=self.trainer,
            preprocessor=self.preprocessor,
        )
