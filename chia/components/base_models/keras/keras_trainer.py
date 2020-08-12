import abc
import random
import time

import numpy as np
import tensorflow as tf

from chia import components, instrumentation
from chia.components.base_models.keras import keras_featureextractor, keras_preprocessor
from chia.helpers import batches_from_pair, make_generator_faster


class KerasTrainer(abc.ABC):
    @abc.abstractmethod
    def observe_inner(self, samples, gt_resource_id, progress_callback=None):
        pass

    @abc.abstractmethod
    def rehearse(self, steps, progress_callback=None):
        pass


class KerasFastSingleShotTrainer(KerasTrainer, instrumentation.Observable):
    def __init__(
        self,
        feature_extractor: keras_featureextractor.KerasFeatureExtractor,
        preprocessor: keras_preprocessor.KerasPreprocessor,
        classifier,
        learning_rate_schedule,
        optimizer,
        batch_size,
        inner_steps,
        sequential_training_batches=1,
    ):
        instrumentation.Observable.__init__(self)

        # Feature Extractor
        self.feature_extractor_new = feature_extractor
        self.preprocessor = preprocessor

        # Classifier
        self.classifier = classifier

        # Learning rate schedule
        self.learning_rate_schedule = learning_rate_schedule

        # Optimization
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.sequential_training_batches = sequential_training_batches
        self._inner_steps = inner_steps

        # State here
        self.current_step = 0
        self._already_observed = False

    def perform_single_gradient_step(self, batch_elements_X, batch_elements_y):
        if self.feature_extractor_new.trainable:
            total_trainable_variables = (
                self.feature_extractor_new.feature_extractor.trainable_variables
                + self.classifier.trainable_variables()
            )
        else:
            total_trainable_variables = self.classifier.trainable_variables()

        # Forward step
        inner_bs = self.batch_size
        acc_gradients = None
        acc_hc_loss = 0
        inner_batch_count = 0

        for inner_batch_X, inner_batch_y in batches_from_pair(
            batch_elements_X, batch_elements_y, inner_bs
        ):
            # Build batch
            batch_X = self.preprocessor.preprocess_image_batch(
                np.stack(inner_batch_X, axis=0), is_training=True
            )
            batch_y = inner_batch_y

            # No numpy stacking here, these could be
            # strings or something else (concept uids)

            with tf.GradientTape() as tape:
                feature_batch = self.feature_extractor_new.feature_extractor(
                    batch_X, training=self.feature_extractor_new.trainable
                )
                hc_loss = self.classifier.loss(
                    feature_batch, batch_y, self.current_step
                )

                if self.feature_extractor_new.trainable:
                    reg_loss = sum(
                        self.feature_extractor_new.feature_extractor.losses
                        + self.classifier.regularization_losses()
                    )
                else:
                    reg_loss = self.classifier.regularization_losses()

                total_loss = hc_loss + reg_loss

            # Backward step
            gradients = tape.gradient(total_loss, total_trainable_variables)
            if acc_gradients is None:
                acc_gradients = gradients
            else:
                acc_gradients = [
                    acc_gradient + new_gradient
                    for acc_gradient, new_gradient in zip(acc_gradients, gradients)
                ]

            acc_hc_loss += hc_loss
            inner_batch_count += 1

        # Optimize
        self.optimizer.learning_rate = self.learning_rate_schedule(self.current_step)
        self.optimizer.apply_gradients(
            zip(
                [
                    acc_gradient / float(inner_batch_count)
                    for acc_gradient in acc_gradients
                ],
                total_trainable_variables,
            )
        )

        self.current_step += 1
        return acc_hc_loss / float(inner_batch_count)

    def observe_inner(self, samples, gt_resource_id, progress_callback=None):
        assert not self._already_observed, "This model can not learn continually"
        assert len(samples) > 0

        total_bs = self.batch_size * self.sequential_training_batches

        def my_gen():
            for inner_step in range(self._inner_steps):
                batch_samples = random.choices(samples, k=total_bs)
                batch_elements_y = []
                batch_elements_X = []
                for sample in batch_samples:
                    batch_elements_X.append(_get_input_img_np(sample))
                    batch_elements_y.append(sample.get_resource(gt_resource_id))

                yield inner_step, (batch_elements_X, batch_elements_y)

        if progress_callback is not None:
            progress_callback(0.0)

        hc_loss_running = 0.0
        time_per_step_running = 0.0
        hc_loss_factor = 0.0
        last_step_end_time = time.time()
        for inner_step, (X, y) in make_generator_faster(
            my_gen, "threading", observable=self, max_buffer_size=20
        ):
            if progress_callback is not None:
                progress_callback(inner_step / float(self._inner_steps))

            hc_loss = self.perform_single_gradient_step(X, y)

            hc_loss_running += hc_loss.numpy()
            hc_loss_factor += 1.0

            step_end_time = time.time()
            time_per_step_running += step_end_time - last_step_end_time

            if self.current_step % 10 == 9:
                self.report_metric(
                    "loss_ravg", hc_loss_running / hc_loss_factor, self.current_step
                )
                self.report_metric(
                    "time_per_step",
                    time_per_step_running / hc_loss_factor,
                    self.current_step,
                )
                hc_loss_running = 0.0
                hc_loss_factor = 0.0
                time_per_step_running = 0.0

            last_step_end_time = step_end_time

        if progress_callback is not None:
            progress_callback(1.0)

    def rehearse(self, steps, progress_callback=None):
        raise ValueError("Cannot learn continually!")


def _get_input_img_np(sample):
    return sample.get_resource("input_img_np")


class KerasTrainerFactory(components.Factory):
    name_to_class_mapping = {"fast_single_shot": KerasFastSingleShotTrainer}
    default_section = "keras_trainer"
