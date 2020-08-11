import networkx as nx
from networkx.algorithms import shortest_paths

from chia import knowledge
from chia.components.evaluators.evaluator import Evaluator


class HierarchicalEvaluator(Evaluator):
    def __init__(self, kb: knowledge.KnowledgeBase):
        self.kb = kb
        self.sample_count = 0
        self.running_cm = dict()

    def reset(self):
        self.sample_count = 0
        self.running_cm = dict()

    def update(self, samples, gt_resource_id, prediction_resource_id):
        for sample in iter(samples):
            gt_uid = sample.get_resource(gt_resource_id)
            pred_uid = sample.get_resource(prediction_resource_id)

            # Confusion Matrix
            if (gt_uid, pred_uid) in self.running_cm.keys():
                self.running_cm[(gt_uid, pred_uid)] += 1
            else:
                self.running_cm[(gt_uid, pred_uid)] = 1

            self.sample_count += 1

    def result(self):
        rgraph = self.kb.get_hyponymy_relation_rgraph()
        root = next(nx.topological_sort(rgraph))

        # Semantic Distance
        running_distance = 0
        running_correct_depth = 0
        running_error_depth = 0
        running_hierarchical_precision = 0
        running_hierarchical_recall = 0

        sample_count_confused = 0
        ugraph = self.kb.get_hyponymy_relation().ugraph()
        for (gt_uid, pred_uid), count in self.running_cm.items():
            lca = nx.lowest_common_ancestor(rgraph, gt_uid, pred_uid)
            lca_depth = nx.shortest_path_length(rgraph, root, lca) + 1

            if gt_uid != pred_uid:
                running_distance += shortest_paths.shortest_path_length(
                    ugraph, gt_uid, pred_uid
                )
                sample_count_confused += count

                # Error Depth in Hierarchy

                running_error_depth += lca_depth * count

            running_correct_depth += lca_depth * count

            # Hierarchical Precision / Recall
            gt_ancestors = set(nx.ancestors(rgraph, gt_uid)) | {gt_uid}
            pred_ancestors = set(nx.ancestors(rgraph, pred_uid)) | {pred_uid}

            hierarchical_precision = len(
                pred_ancestors.intersection(gt_ancestors)
            ) / float(len(pred_ancestors))
            hierarchical_recall = len(
                pred_ancestors.intersection(gt_ancestors)
            ) / float(len(gt_ancestors))

            running_hierarchical_precision += hierarchical_precision * count
            running_hierarchical_recall += hierarchical_recall * count

        # Cannot use tuples as keys in json...
        nested_cm = dict()
        for (gt_uid, pred_uid), count in self.running_cm.items():
            if gt_uid not in nested_cm.keys():
                nested_cm[gt_uid] = dict()

            nested_cm[gt_uid][pred_uid] = count

        ret_dict = dict()
        if self.sample_count > 0:
            ret_dict.update(
                {
                    "semantic_distance": float(running_distance)
                    / float(self.sample_count),
                    "abs_confusion_matrix": nested_cm,
                    "correct_depth": float(running_correct_depth)
                    / float(self.sample_count),
                    "hierarchical_recall": float(running_hierarchical_recall)
                    / float(self.sample_count),
                    "hierarchical_precision": float(running_hierarchical_precision)
                    / float(self.sample_count),
                    "accuracy_from_cm": float(self.sample_count - sample_count_confused)
                    / float(self.sample_count),
                }
            )
        if sample_count_confused > 0:
            ret_dict.update(
                {
                    "semantic_distance_confused": float(running_distance)
                    / float(sample_count_confused),
                    "error_depth": float(running_error_depth)
                    / float(sample_count_confused),
                }
            )

        return ret_dict
