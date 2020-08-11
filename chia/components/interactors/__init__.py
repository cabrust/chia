from chia import components
from chia.components.interactors import noisy_oracle, oracle


class InteractorFactory(components.Factory):
    name_to_class_mapping = {
        "oracle": oracle.OracleInteractor,
        "noisy_oracle": noisy_oracle.NoisyOracleInteractor,
    }
