import time

from chia.components.runners.runner import Runner


class EpochRunner(Runner):
    def __init__(self, experiment_container, epochs, max_test_samples=None):
        super().__init__(experiment_container)
        self.epochs = epochs
        self.max_test_samples = max_test_samples

    def run(self):
        # Get out some members of container
        dataset = self.experiment_container.dataset
        base_model = self.experiment_container.base_model

        # Build training data
        self.log_info("Loading training pool 0...")
        training_samples = dataset.train_pool(0, "label_gt")

        # Build test data
        self.log_info("Loading testing pool 0...")
        if self.max_test_samples is not None:
            test_samples = dataset.test_pool(0, "label_gt")[: self.max_test_samples]
        else:
            test_samples = dataset.test_pool(0, "label_gt")

        # "Interact"
        self.log_info("Performing interaction...")
        training_samples = self.experiment_container.interactor.query_annotations_for(
            training_samples, "label_gt", "label_ann"
        )

        for epoch in range(self.epochs):
            epoch_begin_time = time.time()

            self.log_info(f"Start of epoch {epoch + 1} of {self.epochs}")
            self.log_info("Observing training data...")
            base_model.observe(training_samples, "label_ann")

            self.log_info("Predicting test data...")
            predicted_test_samples = base_model.predict(test_samples, "label_pred")

            # Go over all evaluators
            self.log_info("Evaluating predicted test data...")
            result_dict = {}
            for evaluator in self.experiment_container.evaluators:
                evaluator.update(predicted_test_samples, "label_gt", "label_pred")
                result_dict.update(evaluator.result())
                evaluator.reset()

            # Store epoch duration
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_begin_time
            result_dict.update({"epoch_duration": epoch_duration})

            self.report_result(result_dict, step=epoch + 1)
            self.log_info("Epoch done.")
