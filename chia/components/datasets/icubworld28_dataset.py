import os

import imageio

from chia import data
from chia.components.datasets import dataset
from chia.knowledge import relation

_namespace_uid = "iCubWorld28"

_icubworld28_labels_to_wordnet = [
    ("cup", "cup.n.01"),
    ("dishwashing-detergent", "dishwasher_detergent.n.01"),
    ("laundry-detergent", "laundry_detergent.n.01"),
    ("plate", "plate.n.04"),
    ("soap", "bar_soap.n.01"),
    ("sponge", "sponge.n.01"),
    ("sprayer", "atomizer.n.01"),
]


class iCubWorld28Dataset(dataset.Dataset):
    def __init__(self, base_path):
        self.base_path = base_path

    def setup(self, **kwargs):
        pass

    def train_pool_count(self):
        return 4

    def test_pool_count(self):
        return 4

    def train_pool(self, index, label_resource_id):
        return self.get_train_pool_for(index + 1, label_resource_id)

    def test_pool(self, index, label_resource_id):
        return self.get_test_pool_for(index + 1, label_resource_id)

    def namespace(self):
        return _namespace_uid

    def get_train_pool_for(self, day, label_resource_id):
        return self._build_samples("train", day, label_resource_id)

    def get_test_pool_for(self, day, label_resource_id):
        return self._build_samples("test", day, label_resource_id)

    def _build_samples(self, split, day, label_resource_id):
        assert day > 0
        samples = []
        for (category, wordnet_synset) in _icubworld28_labels_to_wordnet:
            for individual in range(1, 4 + 1):
                sample_dir = os.path.join(
                    self.base_path,
                    split,
                    f"day{day}",
                    category,
                    f"{category}{individual}",
                )
                for filename in sorted(os.listdir(sample_dir)):
                    samples += [
                        data.Sample(
                            source=self.__class__.__name__,
                            uid=f"{_namespace_uid}::{split}:{day}:"
                            + f"{category}{individual}:{filename}",
                        )
                        .add_resource(
                            self.__class__.__name__,
                            label_resource_id,
                            f"{_namespace_uid}::{category}{individual}",
                        )
                        .add_resource(
                            self.__class__.__name__,
                            "image_input_np",
                            imageio.imread(os.path.join(sample_dir, filename)),
                        )
                    ]
        return samples

    def get_hyponymy_relation_source(self):
        relation_ = []
        for label, synset in _icubworld28_labels_to_wordnet:
            for individual in range(1, 4 + 1):
                relation_ += [
                    (f"{_namespace_uid}::{label}{individual}", f"WordNet3.0::{synset}")
                ]
        return relation.StaticRelationSource(relation_)

    def prediction_targets(self):
        return [
            f"{_namespace_uid}::{concept_uid}{individual}"
            for individual in range(1, 4 + 1)
            for concept_uid, _ in _icubworld28_labels_to_wordnet
        ]
