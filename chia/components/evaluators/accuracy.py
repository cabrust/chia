from chia.components.evaluators.evaluator import Evaluator


class AccuracyEvaluator(Evaluator):
    def __init__(self, kb):
        self.correct_count = 0
        self.sample_count = 0

    def reset(self):
        self.correct_count = 0
        self.sample_count = 0

    def update(self, samples, gt_resource_id, prediction_resource_id):
        for sample in iter(samples):
            self.sample_count += 1
            if sample.get_resource(gt_resource_id) == sample.get_resource(
                prediction_resource_id
            ):
                self.correct_count += 1

    def result(self):
        return {"accuracy": float(self.correct_count) / float(self.sample_count)}
