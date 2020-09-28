import json
import pathlib

import numpy as np
from PIL import Image

from chia import data, instrumentation, knowledge
from chia.components.datasets import dataset
from chia.helpers import NetworkResistantImage
from chia.knowledge import relation


class JSONDataset(dataset.Dataset, instrumentation.Observable):
    def __init__(self, json_path, base_path, side_length, use_lazy_mode=True):
        instrumentation.Observable.__init__(self)

        self._json_path = json_path
        self._base_path = base_path
        self._side_length = side_length
        self._use_lazy_mode = use_lazy_mode

        # Read JSON file
        with open(self._json_path) as json_file:
            self._json_data = json.load(json_file)

        # Check format
        if self._json_data["file_format"] != "JSONDataset.v1":
            raise ValueError("Unsupported JSON dataset file format!")

        # Knowledge
        self._prediction_targets = set(self._json_data["prediction_targets"])
        self._hyponymy_relation_data = self._json_data["hyponymy_relation"]
        self._namespace = self._json_data["namespace"]

        # Samples
        self._train_pools = self._json_data["train_pools"]
        self._test_pools = self._json_data["test_pools"]

    def setup(self, **kwargs):
        pass

    def train_pool_count(self):
        return len(self._test_pools)

    def test_pool_count(self):
        return len(self._train_pools)

    def train_pool(self, index, label_resource_id):
        return [
            self._build_sample(sample_dict, label_resource_id)
            for sample_dict in self._train_pools[index]
        ]

    def test_pool(self, index, label_resource_id):
        return [
            self._build_sample(sample_dict, label_resource_id)
            for sample_dict in self._test_pools[index]
        ]

    def namespace(self):
        return self._namespace

    def get_hyponymy_relation_source(self) -> knowledge.RelationSource:
        return relation.StaticRelationSource(self._hyponymy_relation_data)

    def prediction_targets(self):
        return self._prediction_targets

    def _build_sample(self, sample_dict: dict, label_resource_id: str) -> data.Sample:
        image_location = pathlib.Path(self._base_path) / sample_dict["image_location"]
        uid = sample_dict["uid"]
        label_gt = sample_dict["label_gt"]
        sample = data.Sample(source=self.__class__.__name__, uid=uid)
        sample = sample.add_resource(
            self.__class__.__name__, label_resource_id, label_gt
        )

        if self._use_lazy_mode:
            sample = sample.add_resource(
                self.__class__.__name__,
                "image_location",
                image_location,
            ).add_lazy_resource(
                self.__class__.__name__, "input_img_np", self._load_from_location
            )
        else:
            sample = sample.add_resource(
                self.__class__.__name__,
                "image_location",
                image_location,
            )
            sample = sample.add_resource(
                self.__class__.__name__,
                "input_img_np",
                self._load_from_location(sample),
            )

        return sample

    def _load_from_location(self, sample_):
        im = NetworkResistantImage.open(
            sample_.get_resource("image_location"), self
        ).resize((self._side_length, self._side_length), Image.ANTIALIAS)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im)
