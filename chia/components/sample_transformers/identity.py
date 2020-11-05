from chia.components.sample_transformers.sample_transformer import SampleTransformer


class IdentitySampleTransformer(SampleTransformer):
    def transform(self, samples):
        return samples
