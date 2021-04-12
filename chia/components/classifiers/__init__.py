from chia import components
from chia.components.classifiers import (
    keras_idk_hc,
    keras_labelsharing_hc,
    keras_onehot_hc,
)


class ClassifierFactory(components.Factory):
    name_to_class_mapping = {
        "keras_idk": keras_idk_hc.IDKEmbeddingBasedKerasHC,
        "keras_labelsharing": keras_labelsharing_hc.LabelSharingEmbeddingBasedKerasHC,
        "keras_onehot": keras_onehot_hc.OneHotEmbeddingBasedKerasHC,
    }
    default_section = "classifier"
