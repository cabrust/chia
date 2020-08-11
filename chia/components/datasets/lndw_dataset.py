import csv
import os

import numpy as np
from PIL import Image

from chia import data
from chia.components.datasets import dataset
from chia.knowledge import relation

_namespace_uid = "LNDW"


class LNDWDataset(dataset.Dataset):
    def __init__(self, base_path, side_length, viability_threshold=3.0):
        self.base_path = base_path
        self.side_length = side_length
        self._viability_threshold = viability_threshold

        self.all_classes_even_unviable = []
        with open(os.path.join(self.base_path, "classes.csv")) as classes_file:
            reader = csv.reader(classes_file, delimiter=";")
            header = next(reader)
            fields = {
                "folder": "ID",
                "class_name": "Class Name",
                "individual_id": "No.",
                "grade": "Grade",
                "wordnet": "WordNet",
            }
            fields = {k: header.index(v) for k, v in fields.items()}
            for line in reader:
                self.all_classes_even_unviable += [
                    {k: line[v] for k, v in fields.items()}
                ]

            self.viable_classes = [
                class_
                for class_ in self.all_classes_even_unviable
                if self._viable(class_)
            ]

        self.wordnet_mapping = []
        for class_ in self.viable_classes:
            self.wordnet_mapping += [
                (
                    f"{_namespace_uid}::{class_['class_name']}"
                    + f"{int(class_['individual_id']):02d}",
                    f"WordNet3.0::{class_['wordnet']}",
                )
            ]
            self.wordnet_mapping += [
                (
                    f"{_namespace_uid}::{class_['class_name']}",
                    f"WordNet3.0::{class_['wordnet']}",
                )
            ]

        # Attributes are set in setup()
        self.individuals = None
        self.setup()

    def setup(self, individuals=False, **kwargs):
        self.individuals = individuals

    def setups(self):
        return [{"individuals": True}, {"individuals": False}]

    def train_pool_count(self):
        return 1

    def test_pool_count(self):
        return 1

    def train_pool(self, index, label_resource_id):
        assert index == 0
        return self.get_train_pool(label_resource_id, self.individuals)

    def test_pool(self, index, label_resource_id):
        assert index == 0
        return self.get_test_pool(label_resource_id, self.individuals)

    def namespace(self):
        return _namespace_uid

    def _viable(self, class_):
        return float(class_["grade"]) <= self._viability_threshold

    def _filenames(self):
        return [
            "01W.jpg",
            "02SW.jpg",
            "03S.jpg",
            "04SE.jpg",
            "05E.jpg",
            "06NE.jpg",
            "07N.jpg",
            "08NW.jpg",
            "09TOP.jpg",
        ]

    def _build_sample(self, class_, filename, label_resource_id, individuals):
        # Open and resize image
        the_image = Image.open(
            os.path.join(self.base_path, f"{int(class_['folder']):02d}", filename)
        )
        the_image = np.asarray(
            the_image.resize((self.side_length, self.side_length), Image.ANTIALIAS)
        )

        if individuals:
            label_string = (
                f"{_namespace_uid}::{class_['class_name']}"
                + f"{int(class_['individual_id']):02d}"
            )
        else:
            label_string = f"{_namespace_uid}::{class_['class_name']}"

        # Build sample
        the_sample = (
            data.Sample(
                source=self.__class__.__name__,
                uid=f"{_namespace_uid}::{class_}.{filename}",
            )
            .add_resource(self.__class__.__name__, "input_img_np", the_image)
            .add_resource(self.__class__.__name__, label_resource_id, label_string)
        )
        return the_sample

    def get_train_pool(self, label_resource_id, individuals=False):
        samples = []
        for class_ in self.viable_classes:
            for filename in self._filenames()[:8]:
                the_sample = self._build_sample(
                    class_, filename, label_resource_id, individuals
                )
                samples += [the_sample]

        return samples

    def get_test_pool(self, label_resource_id, individuals=False):
        samples = []
        for class_ in self.viable_classes:
            for filename in self._filenames()[8:]:
                the_sample = self._build_sample(
                    class_, filename, label_resource_id, individuals
                )
                samples += [the_sample]

        return samples

    def get_hyponymy_relation_source(self):
        return relation.StaticRelationSource(self.wordnet_mapping)

    def prediction_targets(self):
        if self.individuals:
            return {
                f"{_namespace_uid}::{class_['class_name']}"
                + f"{int(class_['individual_id']):02d}"
                for class_ in self.viable_classes
            }
        else:
            return {
                f"{_namespace_uid}::{class_['class_name']}"
                for class_ in self.viable_classes
            }
