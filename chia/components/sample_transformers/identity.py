from chia.components.sample_transformers.sample_transformer import SampleTransformer


class IdentitySampleTransformer(SampleTransformer):
    def transform(self, samples, is_training: bool, label_resource_id: str):
        return samples
