from chia import components
from chia.components.classifiers import keras_idk_hc, keras_onehot_hc


class ClassifierFactory(components.Factory):
    name_to_class_mapping = {
        "keras_onehot": keras_onehot_hc.OneHotEmbeddingBasedKerasHC,
        "keras_idk": keras_idk_hc.IDKEmbeddingBasedKerasHC,
    }
    default_section = "classifier"
