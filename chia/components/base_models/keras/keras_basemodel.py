import multiprocessing
import pickle as pkl
import time

import numpy as np

from chia import components, instrumentation
from chia.components.base_models.incremental_model import ProbabilityOutputModel
from chia.components.base_models.keras import (
    keras_featureextractor,
    keras_preprocessor,
    keras_trainer,
)
from chia.components.classifiers import keras_hierarchicalclassification
from chia.helpers import batches_from, make_generator_faster


class KerasBaseModel(ProbabilityOutputModel, instrumentation.Observable):
    def __init__(
        self,
        classifier: keras_hierarchicalclassification.KerasHierarchicalClassifier,
        feature_extractor: keras_featureextractor.KerasFeatureExtractor,
        preprocessor: keras_preprocessor.KerasPreprocessor,
        trainer: keras_trainer.KerasTrainer,
        batch_size,
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
            pool = multiprocessing.pool.ThreadPool(4)
            for small_batch_ in batches_from(samples, batch_size=batch_size):
                built_image_batch_ = self.build_image_batch(small_batch_, pool)
                yield small_batch_, built_image_batch_
            pool.close()

        tp_before_data = time.time()
        faster_generator = make_generator_faster(
            my_gen, method="threading", observable=self, max_buffer_size=50
        )
        for (small_batch, built_image_batch) in faster_generator:
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
            pool = multiprocessing.pool.ThreadPool(4)
            for small_batch_ in batches_from(samples, batch_size=batch_size):
                built_image_batch_ = self.build_image_batch(small_batch_, pool)
                yield small_batch_, built_image_batch_

        for small_batch, built_image_batch in make_generator_faster(
            my_gen, method="synchronous", observable=self, max_buffer_size=5
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
        with open(path + "_ilstate.pkl", "wb") as target:
            pkl.dump(self.current_step, target)
        self.classifier.save(path)

    def restore(self, path):
        self.feature_extractor_new.feature_extractor.load_weights(path + "_features.h5")
        with open(path + "_ilstate.pkl", "rb") as target:
            self.current_step = pkl.load(target)
        self.classifier.restore(path)

    def build_image_batch(self, samples, pool=None):
        assert len(samples) > 0
        if pool is not None:
            np_images = pool.map(_get_input_img_np, samples)
        else:
            np_images = [sample.get_resource("input_img_np") for sample in samples]
        return np.stack(np_images, axis=0)


def _get_input_img_np(sample):
    return sample.get_resource("input_img_np")


class KerasBaseModelFactory(components.Factory):
    name_to_class_mapping = KerasBaseModel
    default_section = "keras_base_model"
