import json
import os

import numpy as np
from PIL import Image

from chia import data, instrumentation
from chia.components.datasets import dataset
from chia.helpers import NetworkResistantImage
from chia.knowledge import relation

_namespace_uid = "iNaturalist2018"


class iNaturalist2018Dataset(dataset.Dataset, instrumentation.Observable):
    def setup(self, **kwargs):
        pass

    def train_pool_count(self):
        return 1

    def test_pool_count(self):
        return 1

    def train_pool(self, index, label_resource_id):
        assert index == 0
        return self.pool_from_json("train2018.json", label_resource_id)

    def test_pool(self, index, label_resource_id):
        assert index == 0
        return self.pool_from_json("val2018.json", label_resource_id)

    def namespace(self):
        return _namespace_uid

    def __init__(self, base_path, side_length):
        instrumentation.Observable.__init__(self)
        self.base_path = base_path
        self.side_length = side_length

        self._id_to_class = dict()
        self._hypernymy_relation = set()

        def is_a(concept, superconcept):
            relation_element = (
                f"{_namespace_uid}::{concept}",
                f"{_namespace_uid}::{superconcept}",
            )
            self._hypernymy_relation |= {relation_element}

        with open(os.path.join(self.base_path, "categories.json")) as json_file:
            json_data = json.load(json_file)
            for json_datum in json_data:
                self._id_to_class[
                    json_datum["id"]
                ] = f"{_namespace_uid}::{json_datum['name']}"

                # Biology!
                is_a(f"Ki{json_datum['kingdom']}", "Root")
                is_a(f"Ph{json_datum['phylum']}", f"Ki{json_datum['kingdom']}")
                is_a(f"Cl{json_datum['class']}", f"Ph{json_datum['phylum']}")
                is_a(f"Or{json_datum['order']}", f"Cl{json_datum['class']}")
                is_a(f"Fa{json_datum['family']}", f"Or{json_datum['order']}")
                is_a(f"Ge{json_datum['genus']}", f"Fa{json_datum['family']}")
                is_a(f"{json_datum['name']}", f"Ge{json_datum['genus']}")

    def pool_from_json(self, filename, label_resource_id):
        with open(os.path.join(self.base_path, filename)) as json_file:
            json_data = json.load(json_file)
        image_list = json_data["images"]
        annotation_list = json_data["annotations"]
        annotations = {ann["image_id"]: ann["category_id"] for ann in annotation_list}

        return [
            self.build_sample(image_dict, label_resource_id, annotations)
            for image_dict in image_list
        ]

    def build_sample(self, image_dict, label_resource_id, annotations):
        image_filename = image_dict["file_name"]
        image_id = image_dict["id"]
        sample_ = data.Sample(
            source=self.__class__.__name__, uid=f"{_namespace_uid}::{image_filename}"
        )
        sample_ = sample_.add_resource(
            self.__class__.__name__,
            label_resource_id,
            self._id_to_class[annotations[image_id]],
        )
        sample_ = sample_.add_resource(
            self.__class__.__name__,
            "image_location",
            os.path.join(self.base_path, image_filename),
        ).add_lazy_resource(
            self.__class__.__name__, "input_img_np", self._load_from_location
        )

        return sample_

    def get_hyponymy_relation_source(self):
        return relation.StaticRelationSource(list(self._hypernymy_relation))

    def _load_from_location(self, sample_):
        im = NetworkResistantImage.open(
            sample_.get_resource("image_location"), self
        ).resize((self.side_length, self.side_length), Image.ANTIALIAS)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im)

    def prediction_targets(self):
        return set(self._id_to_class.values())
