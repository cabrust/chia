import pathlib
import urllib

import numpy as np
from PIL import Image

from chia import data, instrumentation
from chia.components.datasets import dataset
from chia.helpers import NetworkResistantImage
from chia.knowledge import relation

_namespace_uid = "CUB2002011"

HIERARCHY_BASE_URL = (
    "https://raw.githubusercontent.com/cvjena/semantic-embeddings/master/CUB-Hierarchy/"
)


class CUB2002011Dataset(dataset.Dataset, instrumentation.Observable):
    def __init__(self, base_path, side_length, use_lazy_mode=True):
        instrumentation.Observable.__init__(self)

        # Configuration
        self._side_length = side_length
        self._use_lazy_mode = use_lazy_mode

        # Paths and Files
        self._base_path = pathlib.Path(base_path)
        if not self._base_path.exists():
            raise ValueError(f"Invalid base path {self._base_path}, does not exist!")

        # Images
        self._images_txt = self._base_path / "images.txt"
        if not self._images_txt.exists():
            raise ValueError(
                f"Invalid base path {self._base_path}, could not find images.txt!"
            )

        with open(self._images_txt) as images_file:
            split_images = [str(x).strip().split(" ", maxsplit=1) for x in images_file]
            self.images = dict(
                [
                    (int(num), self._base_path / "images" / path)
                    for (num, path) in split_images
                ]
            )

        # Labels
        self._labels_txt = self._base_path / "image_class_labels.txt"
        if not self._labels_txt.exists():
            raise ValueError(
                f"Invalid base path {self._base_path}, could not find image_class_labels.txt!"
            )

        with open(self._labels_txt) as labels_file:
            stripped_labels = [
                str(x).strip().split(" ", maxsplit=1) for x in labels_file
            ]
            self.labels = dict(
                [(int(num), int(label)) for (num, label) in stripped_labels]
            )

        # Train / Test Split
        self._train_test_split_txt = self._base_path / "train_test_split.txt"
        if not self._train_test_split_txt.exists():
            raise ValueError(
                f"Invalid base path {self._base_path}, could not find train_test_split.txt!"
            )

        with open(self._train_test_split_txt) as train_test_split_file:
            split_train_test_split = [
                str(x).strip().split(" ", maxsplit=1) for x in train_test_split_file
            ]
            self.train_test_split = dict(
                [
                    (int(num), int(traintest))
                    for (num, traintest) in split_train_test_split
                ]
            )

        # Small Sanity Check
        if not set(self.images.keys()) == set(self.labels.keys()):
            raise ValueError("Integrity problem, please redownload dataset.")
        if not set(self.images.keys()) == set(self.train_test_split.keys()):
            raise ValueError("Integrity problem, please redownload dataset.")

        self.setup()

    def setup(self, hierarchy_mode="wikispecies", **kwargs):
        self._hierarchy_mode = hierarchy_mode
        if self._hierarchy_mode not in ("wikispecies", "flat", "balanced"):
            raise ValueError(
                f'{self._hierarchy_mode} is not a valid hierarchy mode! Use "wikispecies", "flat" or "balanced".'
            )

        self._load_hierarchy()

    def setups(self):
        return [
            {"hierarchy_mode": "wikispecies"},
            {"hierarchy_mode": "flat"},
            {"hierarchy_mode": "balanced"},
        ]

    def train_pool_count(self):
        return 1

    def test_pool_count(self):
        return 1

    def train_pool(self, index, label_resource_id):
        return self._make_pool(1, label_resource_id)

    def test_pool(self, index, label_resource_id):
        return self._make_pool(0, label_resource_id)

    def namespace(self):
        return _namespace_uid

    def get_hyponymy_relation_source(self):
        return relation.StaticRelationSource(self.relation)

    def prediction_targets(self):
        return [self._chia_class_name(num + 1) for num in range(200)]

    def _load_hierarchy(self):
        # Class names
        classes_filename = f"classes_{'wikispecies-hierarchy' if self._hierarchy_mode=='wikispecies' else self._hierarchy_mode}.txt"
        classes_url = f"{HIERARCHY_BASE_URL}/{classes_filename}"
        self._classes_txt = self._base_path / classes_filename

        self._attempt_download(self._classes_txt, classes_url)

        with open(self._classes_txt) as classes_file:
            split_classes = [
                str(x).strip().split(" ", maxsplit=1) for x in classes_file
            ]
            self.classes = dict([(int(num), name) for (num, name) in split_classes])

        # Relation
        relation_filename = f"cub_{self._hierarchy_mode}.parent-child.txt"
        relation_url = f"{HIERARCHY_BASE_URL}/{relation_filename}"
        self._relation_txt = self._base_path / relation_filename

        self._attempt_download(self._relation_txt, relation_url)

        with open(self._relation_txt) as relation_file:
            split_relations = [
                str(x).strip().split(" ", maxsplit=1) for x in relation_file
            ]
            self.relation = [
                (self._chia_class_name(int(child)), self._chia_class_name(int(parent)))
                for (parent, child) in split_relations
            ]

    def _attempt_download(self, target: pathlib.Path, url):
        if target.exists():
            return
        self.log_info(f"Downloading missing CUB2002011 data from {url}...")
        with urllib.request.urlopen(url) as response, open(target, "wb") as out_file:
            data = response.read()
            out_file.write(data)

    def _chia_class_name(self, num):
        return f"{_namespace_uid}::{num:1d}{self.classes[num]}"

    def _make_pool(self, traintest, label_resource_id):
        image_ids = [
            image_id
            for image_id in self.images.keys()
            if self.train_test_split[image_id] == traintest
        ]
        tuples = [(image_id, self.labels[image_id]) for image_id in image_ids]

        return [
            self._build_sample(image_id, label_id, label_resource_id, str(traintest))
            for image_id, label_id in tuples
        ]

    def _build_sample(self, image_id, label_id, label_resource_id, split):
        sample_ = data.Sample(
            source=self.__class__.__name__, uid=f"{_namespace_uid}::{split}:{image_id}"
        ).add_resource(
            self.__class__.__name__,
            label_resource_id,
            self._chia_class_name(label_id),
        )
        if self._use_lazy_mode:
            sample_ = sample_.add_resource(
                self.__class__.__name__,
                "image_location",
                self.images[image_id],
            ).add_lazy_resource(
                self.__class__.__name__, "input_img_np", self._load_from_location
            )
        else:
            sample_ = sample_.add_resource(
                self.__class__.__name__,
                "image_location",
                self.images[image_id],
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
        ).resize((self._side_length, self._side_length), Image.ANTIALIAS)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.asarray(im)
