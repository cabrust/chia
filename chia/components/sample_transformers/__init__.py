from chia import components
from chia.components.sample_transformers import identity
from chia.components.sample_transformers.sample_transformer import SampleTransformer


class SampleTransformerFactory(components.Factory):
    name_to_class_mapping = {"identity": identity.IdentitySampleTransformer}


__all__ = ["SampleTransformer", "SampleTransformerFactory"]
