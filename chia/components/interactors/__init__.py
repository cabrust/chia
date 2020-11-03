from chia import components
from chia.components.interactors import oracle


class InteractorFactory(components.Factory):
    name_to_class_mapping = {"oracle": oracle.OracleInteractor}
