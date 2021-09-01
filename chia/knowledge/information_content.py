import abc
import math

import networkx as nx

from chia import components


class InformationContentCalculator(abc.ABC):
    @abc.abstractmethod
    def calculate_information_content(
        self, concept_uid: str, rgraph: nx.DiGraph
    ) -> float:
        pass


class Sanchez2011OriginalICC(InformationContentCalculator):
    def calculate_information_content(self, concept_uid: str, rgraph: nx.DiGraph):
        exclusive_leaves = set(
            filter(
                lambda n: rgraph.out_degree[n] == 0, nx.descendants(rgraph, concept_uid)
            )
        ) - {concept_uid}

        all_leaves = set(filter(lambda n: rgraph.out_degree[n] == 0, rgraph.nodes))

        ancestors = set(nx.ancestors(rgraph, concept_uid)) | {concept_uid}

        index = -math.log(
            ((len(exclusive_leaves) / float(len(ancestors))) + 1.0)
            / (float(len(all_leaves)) + 1.0)
        )

        return math.fabs(index)


class Sanchez2011ModifiedICC(InformationContentCalculator):
    def calculate_information_content(self, concept_uid: str, rgraph: nx.DiGraph):

        all_leaves = set(filter(lambda n: rgraph.out_degree[n] == 0, rgraph.nodes))

        non_exclusive_leaves = (
            set(nx.descendants(rgraph, concept_uid)) | {concept_uid}
        ) & all_leaves

        ancestors = set(nx.ancestors(rgraph, concept_uid)) | {concept_uid}

        index = -math.log(
            ((len(non_exclusive_leaves) / float(len(ancestors))) + 1.0)
            / (float(len(all_leaves)) + 1.0)
        )

        return math.fabs(index)


class Zhou2008ModifiedICC(InformationContentCalculator):
    def calculate_information_content(self, concept_uid: str, rgraph: nx.DiGraph):
        root = next(nx.topological_sort(rgraph))

        all_leaves = set(filter(lambda n: rgraph.out_degree[n] == 0, rgraph.nodes))
        all_leaf_depths = [
            nx.shortest_path_length(rgraph, root, leaf) for leaf in all_leaves
        ]
        highest_depth = max(all_leaf_depths)
        uid_depth = nx.shortest_path_length(rgraph, root, concept_uid)
        descendants = set(nx.descendants(rgraph, concept_uid)) | {concept_uid}

        k = 0.6  # Harispe et al. 2015, page 55 claims that this is the "original" value.
        index1 = 1.0 - (math.log(len(descendants)) / math.log(len(rgraph.nodes)))
        index2 = math.log(uid_depth + 1) / math.log(highest_depth + 1)

        index = k * index1 + (1.0 - k) * index2

        return math.fabs(index)


class InformationContentCalculatorFactory(components.Factory):
    name_to_class_mapping = {
        "sanchez_2011_original": Sanchez2011OriginalICC,
        "sanchez_2011_modified": Sanchez2011ModifiedICC,
        "zhou_2008_modified": Zhou2008ModifiedICC,
    }
