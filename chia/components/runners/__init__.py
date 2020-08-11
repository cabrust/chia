from chia import components
from chia.components.runners import epoch
from chia.components.runners.runner import Runner


class RunnerFactory(components.Factory):
    name_to_class_mapping = {"epoch": epoch.EpochRunner}


__all__ = ["Runner", "RunnerFactory"]
