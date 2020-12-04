import time

from chia.components.runners.runner import Runner


class EpochRunner(Runner):
    def __init__(
        self,
        experiment_container,
        epochs,
        max_test_samples=None,
        load_path=None,
        save_path=None,
        test_chunk_size=512,
    ):
        super().__init__(experiment_container)
        self.epochs = epochs
        self.max_test_samples = max_test_samples
        self.load_path = load_path
        self.save_path = save_path
        self.test_chunk_size = test_chunk_size

    def run(self):
        # Get out some members of container
        dataset = self.experiment_container.dataset
        base_model = self.experiment_container.base_model
        knowledge_base = self.experiment_container.knowledge_base
        sample_transformers = self.experiment_container.sample_transformers

        # Build training data
        self.log_info("Loading training pool 0...")
        training_samples = dataset.train_pool(0, "label_gt")

        # Build test data
        self.log_info("Loading testing pool 0...")
        if self.max_test_samples is not None:
            test_samples = dataset.test_pool(0, "label_gt")[: self.max_test_samples]
        else:
            test_samples = dataset.test_pool(0, "label_gt")

        test_chunks = [
            test_samples[i : i + self.test_chunk_size]
            for i in range(0, len(test_samples), self.test_chunk_size)
        ]

        # Load model if any
        if self.load_path is not None:
            # Restore knowledge base and model
            knowledge_base.restore(self.load_path)
            base_model.restore(self.load_path)

        # "Interact"
        self.log_info("Performing interaction...")
        training_samples = self.experiment_container.interactor.query_annotations_for(
            training_samples, "label_gt", "label_ann"
        )

        # Transform samples
        self.log_info("Performing sample transform...")
        for sample_transformer in sample_transformers:
            training_samples = sample_transformer.transform(
                training_samples, is_training=True, label_resource_id="label_ann"
            )
            test_samples = sample_transformer.transform(
                test_samples, is_training=False, label_resource_id="label_gt"
            )

        for epoch in range(self.epochs):
            epoch_begin_time = time.time()

            self.log_info(f"Start of epoch {epoch + 1} of {self.epochs}")
            self.log_info("Observing training data...")
            base_model.observe(training_samples, "label_ann")

            self.log_info("Predicting test data...")
            for test_chunk in test_chunks:
                predicted_test_chunk = base_model.predict(test_chunk, "label_pred")
                # Go over all evaluators
                for evaluator in self.experiment_container.evaluators:
                    evaluator.update(predicted_test_chunk, "label_gt", "label_pred")

            self.log_info("Evaluating predicted test data...")
            result_dict = {}
            for evaluator in self.experiment_container.evaluators:
                result_dict.update(evaluator.result())
                evaluator.reset()

            # Store epoch duration
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_begin_time
            result_dict.update({"epoch_duration": epoch_duration})

            self.report_result(result_dict, step=epoch + 1)
            self.log_info("Epoch done.")

        # Save model if requested
        if self.save_path is not None:
            # Save knowledge base and model
            knowledge_base.save(self.save_path)
            base_model.save(self.save_path)
