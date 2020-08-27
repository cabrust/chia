import inspect
import typing

from chia import instrumentation


class Factory:
    name_to_class_mapping: typing.Optional[typing.Dict] = None
    default_section: typing.Optional[str] = None
    i_know_that_var_args_are_not_supported = False

    @classmethod
    def create(cls, config: dict, observers=(), **kwargs):
        temp_observable = instrumentation.NamedObservable(cls.__name__)
        for observer in observers:
            temp_observable.register(observer)

        if isinstance(cls.name_to_class_mapping, dict):
            name = config["name"]
            target_class = cls.name_to_class_mapping[name]
            temp_observable.notify(
                instrumentation.ConfigMessage(
                    cls.__name__,
                    f"{cls.__name__}.name",
                    name,
                    source="config_dict",
                )
            )
        else:
            # If mapping is not a dict, interpret as type directly
            target_class = cls.name_to_class_mapping
            name = target_class.__name__

        init_method_signature = inspect.signature(target_class)

        call_spec_kwargs = dict()

        for parameter, param_spec in init_method_signature.parameters.items():
            # Sanity check
            if (
                param_spec.kind == inspect.Parameter.POSITIONAL_ONLY
                or param_spec.kind == inspect.Parameter.VAR_KEYWORD
                or param_spec.kind == inspect.Parameter.VAR_POSITIONAL
            ):
                if not cls.i_know_that_var_args_are_not_supported:
                    raise ValueError(
                        f"Unsupported kind of constructor parameter {parameter}"
                    )
                else:
                    # Skip the unsupported parameters
                    continue

            # Try to find it
            if parameter in kwargs.keys():
                param_value = kwargs[parameter]
            elif parameter in config.keys():
                param_value = config[parameter]
                temp_observable.notify(
                    instrumentation.ConfigMessage(
                        cls.__name__,
                        f"{target_class.__name__}.{parameter}",
                        param_value,
                        source="config_dict",
                    )
                )
            elif f"{name}_userdefaults.{parameter}" in config.keys():
                param_value = config[f"{name}_userdefaults.{parameter}"]
                temp_observable.notify(
                    instrumentation.ConfigMessage(
                        cls.__name__,
                        f"{target_class.__name__}.{parameter}",
                        param_value,
                        source="userdefaults",
                    )
                )
            elif param_spec.default != inspect.Signature.empty:
                param_value = param_spec.default
                temp_observable.notify(
                    instrumentation.ConfigMessage(
                        cls.__name__,
                        f"{target_class.__name__}.{parameter}",
                        param_value,
                        source="default",
                    )
                )

            else:
                raise ValueError(
                    f"Could not find a value for constructor parameter {parameter}"
                )

            call_spec_kwargs[parameter] = param_value

        # Call constructor
        instance = target_class(**call_spec_kwargs)

        # Register observers if possible
        if isinstance(instance, instrumentation.Observable):
            for observer in observers:
                instance.register(observer)

        for observer in observers:
            temp_observable.unregister(observer)

        return instance


class ContainerFactory(Factory):
    @classmethod
    def create(cls, config: dict, **kwargs):
        name = config["name"]
        target_class = cls.name_to_class_mapping[name]
        return target_class(config, **kwargs)
