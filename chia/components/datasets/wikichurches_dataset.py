import json
import math
import os

import networkx as nx
import numpy as np
from PIL import Image

from chia import data, instrumentation
from chia.components.datasets import dataset
from chia.helpers import NetworkResistantImage
from chia.knowledge import relation

_namespace_uid = "WikiChurches"


class WikiChurchesDataset(dataset.Dataset, instrumentation.Observable):
    def __init__(self, base_path, side_length, use_lazy_mode=True):
        instrumentation.Observable.__init__(self)
        self.base_path = base_path
        self.side_length = side_length
        self.use_lazy_mode = use_lazy_mode

    def setup(self, **kwargs):
        # Record the settings

        self._setup_minimum_elements_per_leaf = 4
        self._setup_random_seed = 93130984
        self._setup_val_fraction = 0.25
        self._setup_test_fraction = 0.25

        self._load_style_ids()
        self._load_image_and_church_metadata()
        self._join_image_filenames_and_church_styles()
        self._load_hierarchy()
        self._clean_labels()

        self._construct_split()

    def _load_style_ids(self):
        # Load the "long" names and Wikidata IDs of all styles
        with open(os.path.join(self.base_path, "style_names.txt")) as cls:
            lines = [x.strip() for x in cls]
            tuples = [x.split(sep=" ", maxsplit=1) for x in lines]

            self.uid_for_style_id = {
                k: f"{_namespace_uid}::{k}_{v}" for (k, v) in tuples
            }

            self.style_ids = {k for (k, v) in tuples}

            if len([k for (k, v) in tuples]) != len({k for (k, v) in tuples}):
                raise ValueError("Non-unique IDs found!")

    def _construct_split(self):
        self._image_filenames_per_leaf = {leaf: [] for leaf in self._leaves}
        for image_filename, style_uid in self.clean_image_filenames_with_style_uids:
            if style_uid in self._leaves:
                self._image_filenames_per_leaf[style_uid] += [image_filename]

        leaf_stats = [
            (leaf, len(self._image_filenames_per_leaf[leaf])) for leaf in self._leaves
        ]
        self.log_info("Class balance statistics:")
        for leaf, stat in sorted(leaf_stats, key=lambda a: a[1], reverse=True):
            self.log_info(f"{leaf:60s} {stat:8d}")

        # Filter
        self._prediction_targets = set(
            [
                leaf
                for leaf, elements in leaf_stats
                if elements >= self._setup_minimum_elements_per_leaf
            ]
        )

        # Build splits
        gen = np.random.RandomState(seed=self._setup_random_seed)

        self._tuples_train = []
        self._tuples_val = []
        self._tuples_test = []

        for style_uid in sorted(self._prediction_targets):
            # Shuffle filenames
            image_filenames = self._image_filenames_per_leaf[style_uid]
            shuffled_filenames = list(sorted(image_filenames))
            gen.shuffle(shuffled_filenames)

            # Count elements and make sure we have enough
            elements = len(image_filenames)
            val_count = int(math.ceil(elements * self._setup_val_fraction))
            test_count = int(math.ceil(elements * self._setup_test_fraction))
            train_count = elements - (val_count + test_count)
            assert train_count * val_count * test_count > 0

            # Save
            self._tuples_val += [
                (image_filename, style_uid)
                for image_filename in shuffled_filenames[:val_count]
            ]
            self._tuples_test += [
                (image_filename, style_uid)
                for image_filename in shuffled_filenames[
                    val_count : val_count + test_count
                ]
            ]
            self._tuples_train += [
                (image_filename, style_uid)
                for image_filename in shuffled_filenames[val_count + test_count :]
            ]

        self.log_info(
            f"Without imprecise. Train: {len(self._tuples_train)}, val: {len(self._tuples_val)}, test: {len(self._tuples_test)}"
        )

        # Add imprecise data to training set
        allowed_concepts = set()
        for style_uid in self._prediction_targets:
            allowed_concepts |= nx.ancestors(self._hypernymy_rgraph, style_uid)
        allowed_concepts -= self._prediction_targets

        for image_filename, style_uid in self.clean_image_filenames_with_style_uids:
            if style_uid in allowed_concepts:
                self._tuples_train += [(image_filename, style_uid)]

        self.log_info(
            f"With imprecise. Train: {len(self._tuples_train)}, val: {len(self._tuples_val)}, test: {len(self._tuples_test)}"
        )
        return

    def _clean_labels(self):
        # Test uniqueness of labels
        count_images_total = 0
        count_images_style_not_in_hierarchy = 0

        count_images_no_label = 0
        count_images_one_label = 0
        count_images_multiple_compatible_labels = 0
        count_images_multiple_incompatible_labels = 0

        self.clean_image_filenames_with_style_uids = []

        for image_filename, style_uids in self.image_filenames_with_style_uids_full:
            count_images_total += 1

            # Remove styles that do not occur in the hierarchy
            processed_style_uids = set(style_uids)
            for style_uid in style_uids:
                if style_uid not in self._hypernymy_rgraph.nodes():
                    count_images_style_not_in_hierarchy += 1
                    self.log_debug(
                        f"Image {image_filename} uses style {style_uid} which is not in the hierarchy!"
                    )
                    processed_style_uids.remove(style_uid)

            if len(processed_style_uids) == 1:
                count_images_one_label += 1
                self.clean_image_filenames_with_style_uids += [
                    (image_filename, list(processed_style_uids)[0])
                ]

            elif len(processed_style_uids) < 1:
                count_images_no_label += 1
                self.log_debug(f"Image {image_filename} without label!")
            else:
                # See if presence of multiple labels is due to presence of ancestors
                augmented_style_uids = set()
                for style_uid in processed_style_uids:
                    augmented_style_uids |= set(
                        nx.ancestors(self._hypernymy_rgraph, style_uid)
                    )

                # If it is, use most precise label
                if processed_style_uids == augmented_style_uids:
                    count_images_one_label += 1
                    count_images_multiple_compatible_labels += 1
                    most_precise_label = sorted(
                        processed_style_uids,
                        key=lambda uid: self.shortest_path_length[uid],
                        reverse=True,
                    )[0]
                    self.clean_image_filenames_with_style_uids += [
                        (image_filename, most_precise_label)
                    ]
                # Otherwise, label will not work
                else:
                    count_images_multiple_incompatible_labels += 1
                    self.log_debug(
                        f"Image {image_filename} has multiple incompatible styles!"
                    )

        self.log_info("Import statistics:")
        self.log_info(
            f"Images total:                                      {count_images_total:8d}"
        )
        self.log_info(
            f"Images without eff. label:                         {count_images_no_label:8d}"
        )
        self.log_info(
            f"Images where style is missing in hierarchy:        {count_images_style_not_in_hierarchy:8d}"
        )
        self.log_info(
            f"Images with multiple incomp. labels:               {count_images_multiple_incompatible_labels:8d}"
        )
        self.log_info(
            f"Images with one label:                             {count_images_one_label:8d}"
        )
        self.log_info(
            f"    of these with multiple comp. labels:           {count_images_multiple_compatible_labels:8d}"
        )

    def _load_hierarchy(self):
        with open(os.path.join(self.base_path, "parent_child_rel.txt")) as hie:
            lines = [x.strip() for x in hie]
            tuples = [x.split(sep=" ", maxsplit=1) for x in lines]
            self.hypernymy_relation = [
                (self.uid_for_style_id[v], self.uid_for_style_id[k])
                for (k, v) in tuples
            ]

        children = [child for child, _ in self.hypernymy_relation]
        parents = [parent for _, parent in self.hypernymy_relation]
        roots = [parent for parent in parents if parent not in children]

        if len(roots) != 1:
            self.log_debug(
                "Hierarchy has multiple roots. Fixing hierarchy by adding custom root node."
            )
            for root in roots:
                self.hypernymy_relation += [(root, f"{_namespace_uid}::ROOT")]

        # Precalculate some helpful things
        temporary_graph = nx.DiGraph()
        temporary_graph.add_edges_from(self.hypernymy_relation)
        self._hypernymy_rgraph: nx.DiGraph = temporary_graph.reverse()

        self._root = next(nx.topological_sort(self._hypernymy_rgraph))
        self._leaves = [
            node
            for node in self._hypernymy_rgraph.nodes
            if self._hypernymy_rgraph.out_degree[node] == 0
        ]
        self._shortest_path_lengths = nx.shortest_path_length(
            self._hypernymy_rgraph, source=root
        )

    def _load_image_and_church_metadata(self):
        # Load image metadata (only to get the file names, but could be used for more later on)
        with open(os.path.join(self.base_path, "image_meta.json")) as image_meta_json:
            self.image_meta: dict = json.load(image_meta_json)

        # Load church metadata to obtain the styles associated with it
        with open(os.path.join(self.base_path, "churches.json")) as churches_json:
            self.churches: dict = json.load(churches_json)

    def _join_image_filenames_and_church_styles(self):
        # Associate image filenames and style uids, skipping the churches (performing a JOIN so to speak)
        self.image_filenames_with_style_uids_full = []

        for image_filename, _ in dict(self.image_meta).items():
            church_id = image_filename.split(sep="_", maxsplit=1)[0]
            if church_id in self.churches.keys():
                church_meta = self.churches[church_id]
                style_uids = [
                    self.uid_for_style_id[style_id]
                    for style_id in church_meta["styles"]
                ]
                self.image_filenames_with_style_uids_full += [
                    (image_filename, style_uids)
                ]
            else:
                self.log_debug(
                    f"Church {church_id} is missing in churches.json, skipping {image_filename}!"
                )

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
            self._build_sample(image_filename, style_uid, label_resource_id, "train")
            for image_filename, style_uid in self._tuples_train
        ]

    def get_test_pool(self, label_resource_id):
        return [
            self._build_sample(image_filename, style_uid, label_resource_id, "val")
            for image_filename, style_uid in self._tuples_val
        ]

    def _build_sample(self, image_filename, style_uid, label_resource_id, split):
        sample_ = data.Sample(
            source=self.__class__.__name__,
            uid=f"{_namespace_uid}::{split}:{image_filename}",
        ).add_resource(
            self.__class__.__name__,
            label_resource_id,
            style_uid,
        )
        if self.use_lazy_mode:
            sample_ = sample_.add_resource(
                self.__class__.__name__,
                "image_location",
                os.path.join(self.base_path, "images", image_filename),
            ).add_lazy_resource(
                self.__class__.__name__, "input_img_np", self._load_from_location
            )
        else:
            sample_ = sample_.add_resource(
                self.__class__.__name__,
                "image_location",
                os.path.join(self.base_path, "images", image_filename),
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
        return relation.StaticRelationSource(self.hypernymy_relation)

    def prediction_targets(self):
        return self._prediction_targets
