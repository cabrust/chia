import os

import numpy as np
from PIL import Image

from chia import data, instrumentation
from chia.components.datasets import dataset
from chia.helpers import NetworkResistantImage
from chia.knowledge import relation

_namespace_uid = "NABirds"


class NABirdsDataset(dataset.Dataset, instrumentation.Observable):
    def __init__(self, base_path, side_length, use_lazy_mode=True):
        instrumentation.Observable.__init__(self)
        self.base_path = base_path
        self.side_length = side_length
        self.use_lazy_mode = use_lazy_mode

        with open(os.path.join(self.base_path, "classes.txt")) as cls:
            lines = [x.strip() for x in cls]
            tuples = [x.split(sep=" ", maxsplit=1) for x in lines]
            tuples = [(int(k), str(v)) for (k, v) in tuples]

            self.uid_for_label_id = {
                k: f"{_namespace_uid}::{int(k):03d}{v}" for (k, v) in tuples
            }

            self.nabirds_ids = {k for (k, v) in tuples}

            if len([k for (k, v) in tuples]) != len({k for (k, v) in tuples}):
                raise ValueError("Non-unique IDs found!")

        with open(os.path.join(self.base_path, "image_class_labels.txt")) as lab:
            with open(os.path.join(self.base_path, "train_test_split.txt")) as tts:
                with open(os.path.join(self.base_path, "images.txt")) as iid:
                    lablines = [x.strip() for x in lab]
                    labtuples = [x.split(sep=" ", maxsplit=1) for x in lablines]
                    labtuples = [
                        (str(some_primary_key), int(label))
                        for (some_primary_key, label) in labtuples
                    ]

                    ttslines = [x.strip() for x in tts]
                    ttstuples = [x.split(sep=" ", maxsplit=1) for x in ttslines]
                    ttstuples = [
                        (str(some_primary_key), int(is_train_or_test))
                        for (some_primary_key, is_train_or_test) in ttstuples
                    ]

                    iidlines = [x.strip() for x in iid]
                    iidtuples = [x.split(sep=" ", maxsplit=1) for x in iidlines]
                    iidtuples = [
                        (str(some_primary_key), str(image_path))
                        for (some_primary_key, image_path) in iidtuples
                    ]
                    self.image_location_for_image_id = {k: v for (k, v) in iidtuples}

                    combinedtuples = [a + b for (a, b) in zip(labtuples, ttstuples)]
                    mismatches = [a != c for (a, b, c, d) in combinedtuples]
                    if any(mismatches):
                        raise ValueError("Mismatch between tts and label files!")

                    combinedtuples = [(a, b, d) for (a, b, c, d) in combinedtuples]

                    self._nabirds_training_tuples = [
                        (img, id) for (img, id, tt) in combinedtuples if tt == 1
                    ]
                    self._nabirds_validation_tuples = [
                        (img, id) for (img, id, tt) in combinedtuples if tt == 0
                    ]

        with open(os.path.join(self.base_path, "hierarchy.txt")) as hie:
            lines = [x.strip() for x in hie]
            tuples = [x.split(sep=" ", maxsplit=1) for x in lines]
            self.tuples = [
                (self.uid_for_label_id[int(k)], self.uid_for_label_id[int(v)])
                for (k, v) in tuples
            ]

    def setup(self, **kwargs):
        pass

    def train_pool_count(self):
        return 1

    def test_pool_count(self):
        return 1

    def train_pool(self, index, label_resource_id):
        assert index == 0
        return self.get_train_pool(label_resource_id)

    def test_pool(self, index, label_resource_id):
        assert index == 0
        return self.get_test_pool(label_resource_id)

    def namespace(self):
        return _namespace_uid

    def get_train_pool(self, label_resource_id):
        return [
            self._build_sample(image_id, label_id, label_resource_id, "train")
            for image_id, label_id in self._nabirds_training_tuples
        ]

    def get_test_pool(self, label_resource_id):
        return [
            self._build_sample(image_id, label_id, label_resource_id, "test")
            for image_id, label_id in self._nabirds_validation_tuples
        ]

    def _build_sample(self, image_id, label_id, label_resource_id, split):
        sample_ = data.Sample(
            source=self.__class__.__name__, uid=f"{_namespace_uid}::{split}:{image_id}"
        ).add_resource(
            self.__class__.__name__,
            label_resource_id,
            self.uid_for_label_id[label_id],
        )
        if self.use_lazy_mode:
            sample_ = sample_.add_resource(
                self.__class__.__name__,
                "image_location",
                os.path.join(
                    self.base_path, "images", self.image_location_for_image_id[image_id]
                ),
            ).add_lazy_resource(
                self.__class__.__name__, "input_img_np", self._load_from_location
            )
        else:
            sample_ = sample_.add_resource(
                self.__class__.__name__,
                "image_location",
                os.path.join(
                    self.base_path, "images", self.image_location_for_image_id[image_id]
                ),
            )
            sample_ = sample_.add_resource(
                self.__class__.__name__,
                "input_img_np",
                self._load_from_location(sample_),
            )
        return sample_

    def _load_from_location(self, sample_):
        im = NetworkResistantImage.open(
            sample_.get_resource("image_location"), self
        ).resize((self.side_length, self.side_length), Image.ANTIALIAS)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im)

    def get_hyponymy_relation_source(self):
        return relation.StaticRelationSource(self.tuples)

    def prediction_targets(self):
        label_ids = {label_id for _, label_id in self._nabirds_training_tuples} | {
            label_id for _, label_id in self._nabirds_validation_tuples
        }
        return {self.uid_for_label_id[label_id] for label_id in label_ids}
