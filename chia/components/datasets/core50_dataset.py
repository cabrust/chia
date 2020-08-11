import glob
import os
import pickle as pkl

import numpy as np

from chia import data
from chia.components.datasets import dataset
from chia.knowledge import relation

_namespace_uid = "CORe50"

_labels_to_wordnet = [
    ("plug_adapter1", "plug.n.05"),
    ("plug_adapter2", "plug.n.05"),
    ("plug_adapter3", "plug.n.05"),
    ("plug_adapter4", "plug.n.05"),
    ("plug_adapter5", "plug.n.05"),
    ("mobile_phone1", "cellular_telephone.n.01"),
    ("mobile_phone2", "cellular_telephone.n.01"),
    ("mobile_phone3", "cellular_telephone.n.01"),
    ("mobile_phone4", "cellular_telephone.n.01"),
    ("mobile_phone5", "cellular_telephone.n.01"),
    ("scissor1", "scissors.n.01"),
    ("scissor2", "scissors.n.01"),
    ("scissor3", "scissors.n.01"),
    ("scissor4", "scissors.n.01"),
    ("scissor5", "scissors.n.01"),
    ("light_bulb1", "light_bulb.n.01"),
    ("light_bulb2", "light_bulb.n.01"),
    ("light_bulb3", "light_bulb.n.01"),
    ("light_bulb4", "light_bulb.n.01"),
    ("light_bulb5", "light_bulb.n.01"),
    ("can1", "soda_can.n.01"),
    ("can2", "soda_can.n.01"),
    ("can3", "soda_can.n.01"),
    ("can4", "soda_can.n.01"),
    ("can5", "soda_can.n.01"),
    ("glass1", "spectacles.n.01"),
    ("glass2", "sunglasses.n.01"),
    ("glass3", "sunglasses.n.01"),
    ("glass4", "sunglasses.n.01"),
    ("glass5", "sunglasses.n.01"),
    ("ball1", "ball.n.01"),
    ("ball2", "tennis_ball.n.01"),
    ("ball3", "football.n.02"),
    ("ball4", "ball.n.01"),
    ("ball5", "football.n.02"),
    ("marker1", "highlighter.n.02"),
    ("marker2", "highlighter.n.02"),
    ("marker3", "highlighter.n.02"),
    ("marker4", "highlighter.n.02"),
    ("marker5", "highlighter.n.02"),
    ("cup1", "cup.n.01"),
    ("cup2", "cup.n.01"),
    ("cup3", "cup.n.01"),
    ("cup4", "cup.n.01"),
    ("cup5", "cup.n.01"),
    ("remote_control1", "remote_control.n.01"),
    ("remote_control2", "remote_control.n.01"),
    ("remote_control3", "remote_control.n.01"),
    ("remote_control4", "remote_control.n.01"),
    ("remote_control5", "remote_control.n.01"),
]


class CORe50Dataset(dataset.Dataset):
    def __init__(self, base_path):
        self.base_path = base_path

        self.imgs = np.load(
            os.path.join(self.base_path, "core50_imgs.npy"), mmap_mode="r"
        )

        with open(os.path.join(self.base_path, "paths.pkl"), "rb") as paths_file:
            self.paths = pkl.load(paths_file)

        with open(
            os.path.join(self.base_path, "labels2names.pkl"), "rb"
        ) as labels_to_names_file:
            self.labels_to_names = pkl.load(labels_to_names_file)

        self.path_to_index = {path: index for index, path in enumerate(self.paths)}

        # Attributes are assigned in setup()
        self.scenario = None
        self.run = None
        self.setup()

    def setups(self):
        setups = []
        for scenario in ["ni", "nc", "nic"]:
            for run in range(self.get_run_count(scenario)):
                setups += [{"scenario": scenario, "run": run}]
        return setups

    def setup(self, scenario="nic", run=0, **kwargs):
        self.scenario = scenario
        self.run = run

    def train_pool_count(self):
        return self.get_train_pool_count(self.scenario, self.run)

    def test_pool_count(self):
        return 1

    def train_pool(self, index, label_resource_id):
        return self.get_pool_for(
            self.scenario, self.run, f"train_batch_{index:02d}", label_resource_id
        )

    def test_pool(self, index, label_resource_id):
        assert index == 0
        return self.get_pool_for(self.scenario, self.run, "test", label_resource_id)

    def namespace(self):
        return _namespace_uid

    def get_train_pool_count(self, scenario, run):
        scenario = str(scenario).lower()
        assert scenario in ["ni", "nc", "nic"]
        filelist_filter = os.path.join(
            self.base_path,
            "batches_filelists",
            f"{scenario.upper()}_inc",
            f"run{run:d}",
            "train_batch_*_filelist.txt",
        )

        return len(glob.glob(filelist_filter))

    def get_run_count(self, scenario):
        scenario = str(scenario).lower()
        assert scenario in ["ni", "nc", "nic"]
        filelist_filter = os.path.join(
            self.base_path, "batches_filelists", f"{scenario.upper()}_inc", "run*"
        )

        return len(glob.glob(filelist_filter))

    def get_pool_for(self, scenario, run, batch, label_resource_id):
        # Find the data
        scenario = str(scenario).lower()
        assert scenario in ["ni", "nc", "nic"]

        filelist_path = os.path.join(
            self.base_path,
            "batches_filelists",
            f"{scenario.upper()}_inc",
            f"run{run:d}",
            f"{batch}_filelist.txt",
        )

        # Find appropriate label map
        label_map = self.labels_to_names[scenario][run]

        samples = []

        with open(filelist_path) as filelist:
            for line in filelist:
                path, class_id = line.strip().split(" ")
                samples += [
                    data.Sample(
                        source=self.__class__.__name__, uid=f"{_namespace_uid}::{path}"
                    )
                    .add_resource(
                        self.__class__.__name__,
                        "input_img_np",
                        self.imgs[self.path_to_index[path]],
                    )
                    .add_resource(
                        self.__class__.__name__,
                        label_resource_id,
                        f"{_namespace_uid}::{label_map[int(class_id)]}",
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
