import random

import numpy as np

from chia import data
from chia.components.datasets import dataset
from chia.knowledge import relation

_namespace_uid = "iCIFAR"

_label_names = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

_labels_to_wordnet = [
    ("apple", "apple.n.01"),
    ("aquarium_fish", "freshwater_fish.n.01"),
    ("baby", "baby.n.01"),
    ("bear", "bear.n.01"),
    ("beaver", "beaver.n.07"),
    ("bed", "bed.n.01"),
    ("bee", "bee.n.01"),
    ("beetle", "beetle.n.01"),
    ("bicycle", "bicycle.n.01"),
    ("bottle", "bottle.n.01"),
    ("bowl", "bowl.n.01"),
    ("boy", "male_child.n.01"),
    ("bridge", "bridge.n.01"),
    ("bus", "bus.n.01"),
    ("butterfly", "butterfly.n.01"),
    ("camel", "camel.n.01"),
    ("can", "can.n.01"),
    ("castle", "castle.n.02"),
    ("caterpillar", "caterpillar.n.01"),
    ("cattle", "cattle.n.01"),
    ("chair", "chair.n.01"),
    ("chimpanzee", "chimpanzee.n.01"),
    ("clock", "clock.n.01"),
    ("cloud", "cloud.n.02"),
    ("cockroach", "cockroach.n.01"),
    ("couch", "sofa.n.01"),
    ("crab", "crab.n.01"),
    ("crocodile", "crocodile.n.01"),
    ("cup", "cup.n.01"),
    ("dinosaur", "dinosaur.n.01"),
    ("dolphin", "dolphin.n.02"),
    ("elephant", "elephant.n.01"),
    ("flatfish", "flatfish.n.02"),
    ("forest", "forest.n.01"),
    ("fox", "fox.n.01"),
    ("girl", "female_child.n.01"),
    ("hamster", "hamster.n.01"),
    ("house", "house.n.01"),
    ("kangaroo", "kangaroo.n.01"),
    ("keyboard", "computer_keyboard.n.01"),
    ("lamp", "lamp.n.01"),
    ("lawn_mower", "lawn_mower.n.01"),
    ("leopard", "leopard.n.02"),
    ("lion", "lion.n.01"),
    ("lizard", "lizard.n.01"),
    ("lobster", "lobster.n.02"),
    ("man", "man.n.01"),
    ("maple_tree", "maple.n.02"),
    ("motorcycle", "motorcycle.n.01"),
    ("mountain", "mountain.n.01"),
    ("mouse", "mouse.n.01"),
    ("mushroom", "mushroom.n.02"),
    ("oak_tree", "oak.n.02"),
    ("orange", "orange.n.01"),
    ("orchid", "orchid.n.01"),
    ("otter", "otter.n.02"),
    ("palm_tree", "palm.n.03"),
    ("pear", "pear.n.01"),
    ("pickup_truck", "pickup.n.01"),
    ("pine_tree", "pine.n.01"),
    ("plain", "plain.n.01"),
    ("plate", "plate.n.04"),
    ("poppy", "poppy.n.01"),
    ("porcupine", "porcupine.n.01"),
    ("possum", "opossum.n.02"),
    ("rabbit", "rabbit.n.01"),
    ("raccoon", "raccoon.n.02"),
    ("ray", "ray.n.07"),
    ("road", "road.n.01"),
    ("rocket", "rocket.n.01"),
    ("rose", "rose.n.01"),
    ("sea", "sea.n.01"),
    ("seal", "seal.n.09"),
    ("shark", "shark.n.01"),
    ("shrew", "shrew.n.02"),
    ("skunk", "skunk.n.04"),
    ("skyscraper", "skyscraper.n.01"),
    ("snail", "snail.n.01"),
    ("snake", "snake.n.01"),
    ("spider", "spider.n.01"),
    ("squirrel", "squirrel.n.01"),
    ("streetcar", "streetcar.n.01"),
    ("sunflower", "sunflower.n.01"),
    ("sweet_pepper", "sweet_pepper.n.02"),
    ("table", "table.n.02"),
    ("tank", "tank.n.01"),
    ("telephone", "telephone.n.01"),
    ("television", "television.n.02"),
    ("tiger", "tiger.n.02"),
    ("tractor", "tractor.n.01"),
    ("train", "train.n.01"),
    ("trout", "trout.n.02"),
    ("tulip", "tulip.n.01"),
    ("turtle", "turtle.n.02"),
    ("wardrobe", "wardrobe.n.01"),
    ("whale", "whale.n.02"),
    ("willow_tree", "willow.n.01"),
    ("wolf", "wolf.n.01"),
    ("woman", "woman.n.01"),
    ("worm", "worm.n.01"),
]


class iCIFARDataset(dataset.Dataset):
    def __init__(self):
        import tensorflow as tf

        (
            (self.train_X, self.train_y),
            (self.test_X, self.test_y),
        ) = tf.keras.datasets.cifar100.load_data(label_mode="fine")

        self.sequence_seed = 19219
        self._update_sequence()

        sorting_sequence_train = np.argsort(self.train_y[:, 0], kind="stable")
        self.train_X = self.train_X[sorting_sequence_train]
        self.train_y = self.train_y[sorting_sequence_train]

        # Attributes are assigned in setup()
        self.classes_per_batch = None
        self.setup()

    def setup(self, classes_per_batch=4, **kwargs):
        self.classes_per_batch = classes_per_batch

    def train_pool_count(self):
        return self.get_train_pool_count(self.classes_per_batch)

    def test_pool_count(self):
        return 1

    def train_pool(self, index, label_resource_id):
        return self.get_train_pool_for(index, label_resource_id, self.classes_per_batch)

    def test_pool(self, index, label_resource_id):
        assert index == 0
        return self.get_test_pool(label_resource_id)

    def namespace(self):
        return _namespace_uid

    def _update_sequence(self):
        random.seed(self.sequence_seed)
        self.sequence = random.sample(range(100), 100)

    def get_train_pool_count(self, classes_per_batch):
        return 100 // classes_per_batch

    def get_train_pool_for(self, batch, label_resource_id, classes_per_batch):
        assert (100 % classes_per_batch) == 0
        batch_count = 100 // classes_per_batch
        assert batch < batch_count

        samples = []
        classes_for_pool = self.sequence[
            (batch * classes_per_batch) : (batch + 1) * classes_per_batch
        ]

        # print(f"Retrieving images for classes {classes_for_pool}")
        for class_ in classes_for_pool:
            training_data_range = list(range(500 * class_, 500 * (class_ + 1)))
            class_X = self.train_X[500 * class_ : 500 * (class_ + 1)]
            class_y = self.train_y[500 * class_ : 500 * (class_ + 1)]
            samples += self._build_samples(
                class_X, class_y, training_data_range, label_resource_id, "train"
            )

        return samples

    def get_test_pool(self, label_resource_id):
        return self._build_samples(
            self.test_X, self.test_y, range(0, 10000), label_resource_id, "test"
        )

    def _build_samples(self, X, y, data_range, label_resource_id, prefix):
        assert X.shape[0] == len(data_range)
        samples = []
        for i, data_id in enumerate(data_range):
            class_label = y[i, 0]
            np_image = X[i]
            samples += [
                data.Sample(
                    source=self.__class__.__name__,
                    uid=f"{_namespace_uid}::{prefix}.{data_id}",
                )
                .add_resource(self.__class__.__name__, "input_img_np", np_image)
                .add_resource(
                    self.__class__.__name__,
                    label_resource_id,
                    f"{_namespace_uid}::{_label_names[int(class_label)]}",
                )
            ]
        return samples

    def get_hyponymy_relation_source(self):
        return relation.StaticRelationSource(
            [
                (f"{_namespace_uid}::{label}", f"WordNet3.0::{synset}")
                for label, synset in _labels_to_wordnet
            ]
        )

    def prediction_targets(self):
        return [
            f"{_namespace_uid}::{concept_uid}" for concept_uid, _ in _labels_to_wordnet
        ]
