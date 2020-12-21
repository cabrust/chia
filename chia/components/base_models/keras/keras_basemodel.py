import pickle as pkl
import time

from chia import components, instrumentation
from chia.components.base_models.incremental_model import ProbabilityOutputModel
from chia.components.base_models.keras import (
    keras_featureextractor,
    keras_preprocessor,
    keras_trainer,
)
from chia.components.classifiers import keras_hierarchicalclassification
from chia.helpers import batches_from, threaded_processor


class KerasBaseModel(ProbabilityOutputModel, instrumentation.Observable):
    def __init__(
        self,
        classifier: keras_hierarchicalclassification.KerasHierarchicalClassifier,
        feature_extractor: keras_featureextractor.KerasFeatureExtractor,
        preprocessor: keras_preprocessor.KerasPreprocessor,
        trainer: keras_trainer.KerasTrainer,
        batch_size,
        num_threads_prediction=8,
    ):
        instrumentation.Observable.__init__(self)

        self.classifier = classifier

        # Batch size
        self.batch_size = batch_size

        # Feature Extractor
        self.feature_extractor_new = feature_extractor

        # Preprocessing
        self.preprocessor = preprocessor

        # Trainer
        self.trainer = trainer

        # Settings
        self.num_threads_prediction = num_threads_prediction

    def observe_inner(self, samples, gt_resource_id, progress_callback=None):
        self.trainer.observe_inner(samples, gt_resource_id, progress_callback)

    def observe(self, samples, gt_resource_id, progress_callback=None):
        self.classifier.observe(samples, gt_resource_id)
        self.observe_inner(samples, gt_resource_id, progress_callback)

    def predict(self, samples, prediction_resource_id):
        return_samples = []
        batch_size = self.batch_size

        total_time_data = 0.0
        total_time_preprocess = 0.0
        total_time_features = 0.0
        total_time_cls = 0.0
        total_time_write = 0.0

        def my_gen():
            for small_batch_ in batches_from(samples, batch_size=batch_size):
                yield small_batch_

        tp_before_data = time.time()

        for (small_batch, built_image_batch) in threaded_processor(
            my_gen, _batch_processor, self, num_threads=self.num_threads_prediction
        ):
            tp_before_preprocess = time.time()
            image_batch = self.preprocessor.preprocess_image_batch(
                built_image_batch, is_training=False
            )
            tp_before_features = time.time()
            feature_batch = self.feature_extractor_new(image_batch, training=False)
            tp_before_cls = time.time()
            predictions_dist = self.classifier.predict_dist(feature_batch)
            predictions = [
                sorted(prediction_dist, key=lambda x: x[1], reverse=True)[0][0]
                for prediction_dist in predictions_dist
            ]

            tp_before_write = time.time()
            return_samples += [
                sample.add_resource(
                    self.__class__.__name__, prediction_resource_id, prediction
                ).add_resource(
                    self.__class__.__name__,
                    prediction_resource_id + "_dist",
                    prediction_dist,
                )
                for prediction, prediction_dist, sample in zip(
                    predictions, predictions_dist, small_batch
                )
            ]
            tp_loop_done = time.time()
            total_time_data += tp_before_preprocess - tp_before_data
            total_time_preprocess += tp_before_features - tp_before_preprocess
            total_time_features += tp_before_cls - tp_before_features
            total_time_cls += tp_before_write - tp_before_cls
            total_time_write += tp_loop_done - tp_before_write

            tp_before_data = time.time()

        self.log_debug("Predict done.")
        self.log_debug(f"Time (data): {total_time_data}")
        self.log_debug(f"Time (preprocess): {total_time_preprocess}")
        self.log_debug(f"Time (features): {total_time_features}")
        self.log_debug(f"Time (cls): {total_time_cls}")
        self.log_debug(f"Time (write): {total_time_write}")
        total_time_overall = (
            total_time_data
            + total_time_preprocess
            + total_time_features
            + total_time_cls
            + total_time_write
        )
        self.log_debug(f"Total time: {total_time_overall}")
        return return_samples

    def predict_probabilities(self, samples, prediction_dist_resource_id):
        return_samples = []
        batch_size = self.batch_size

        def my_gen():
            for small_batch_ in batches_from(samples, batch_size=batch_size):
                yield small_batch_

        for small_batch, built_image_batch in threaded_processor(
            my_gen, _batch_processor, self, num_threads=self.num_threads_prediction
        ):
            image_batch = self.preprocessor.preprocess_image_batch(
                built_image_batch, is_training=False
            )
            feature_batch = self.feature_extractor_new(image_batch, training=False)
            predictions = self.classifier.predict_dist(feature_batch)
            return_samples += [
                sample.add_resource(
                    self.__class__.__name__, prediction_dist_resource_id, prediction
                )
                for prediction, sample in zip(predictions, small_batch)
            ]
        return return_samples

    def save(self, path):
        self.feature_extractor_new.feature_extractor.save_weights(path + "_features.h5")
        with open(path + "_trainerstate.pkl", "wb") as target:
            pkl.dump(self.trainer.current_step, target)
        self.classifier.save(path)

    def restore(self, path):
        self.feature_extractor_new.feature_extractor.load_weights(path + "_features.h5")
        with open(path + "_trainerstate.pkl", "rb") as target:
            self.trainer.current_step = pkl.load(target)
        self.classifier.restore(path)


def _batch_processor(batch_samples):
    batch_elements_X = []  # The input image
    for sample in batch_samples:
        batch_elements_X.append(_get_input_img_np(sample))

    return batch_samples, batch_elements_X


def _get_input_img_np(sample):
    return sample.get_resource("input_img_np")


class KerasBaseModelFactory(components.Factory):
    name_to_class_mapping = KerasBaseModel
    default_section = "keras_base_model"
