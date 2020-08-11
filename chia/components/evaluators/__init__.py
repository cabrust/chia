from chia import components
from chia.components.evaluators import accuracy, hierarchical, topk_accuracy
from chia.components.evaluators.evaluator import Evaluator


class EvaluatorFactory(components.Factory):
    name_to_class_mapping = {
        "accuracy": accuracy.AccuracyEvaluator,
        "topk_accuracy": topk_accuracy.TopKAccuracyEvaluator,
        "hierarchical": hierarchical.HierarchicalEvaluator,
    }


__all__ = ["EvaluatorFactory", "Evaluator"]
