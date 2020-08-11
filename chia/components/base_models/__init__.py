from chia import components
from chia.components.base_models import keras


class BaseModelContainerFactory(components.ContainerFactory):
    name_to_class_mapping = {"keras": keras.KerasBaseModelContainer}
    default_section = "base_model"
