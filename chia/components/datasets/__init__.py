from chia import components
from chia.components.datasets import (
    core50_dataset,
    icifar_dataset,
    icubworld28_dataset,
    ilsvrc2012_dataset,
    inaturalist2018_dataset,
    lndw_dataset,
    nabirds_dataset,
)


class DatasetFactory(components.Factory):
    name_to_class_mapping = {
        "core50": core50_dataset.CORe50Dataset,
        "icifar": icifar_dataset.iCIFARDataset,
        "icubworld28": icubworld28_dataset.iCubWorld28Dataset,
        "inaturalist2018": inaturalist2018_dataset.iNaturalist2018Dataset,
        "ilsvrc2012": ilsvrc2012_dataset.ILSVRC2012Dataset,
        "lndw": lndw_dataset.LNDWDataset,
        "nabirds": nabirds_dataset.NABirdsDataset,
    }
    default_section = "dataset"
