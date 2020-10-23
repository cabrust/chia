import os
import typing

from chia.instrumentation import observable
from chia.instrumentation.observers import observer


def setup_environment(observers: typing.List[observer.Observer] = ()):
    obs = observable.NamedObservable("setup_environment")
    for observer_ in observers:
        obs.register(observer_)

    enable_segfault_catching()
    if not validate_gpu_configuration(observers):
        obs.send_shutdown(False)

    setup_tensorflow()


def enable_segfault_catching():
    import faulthandler

    faulthandler.enable()
    return True


def validate_gpu_configuration(observers=()):
    obs = observable.NamedObservable("validate_gpu_configuration")
    for observer_ in observers:
        obs.register(observer_)

    try:
        import os

        from GPUtil import GPUtil

        # Disable GPUS if desired
        if "CHIA_CPU_ONLY" in os.environ.keys():
            if os.environ["CHIA_CPU_ONLY"] == "1":
                obs.log_info(
                    "Requested CPU only operation."
                    "Disabling all GPUS via environment variable."
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                return True

        gpus = GPUtil.getGPUs()
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
            cuda_filter = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            obs.log_info(f"Found CUDA device filter: {cuda_filter}")
            gpus = [gpu for gpu in gpus if str(gpu.id) in cuda_filter]

        available_gpus = []
        for gpu in gpus:
            obs.log_info(f"Found GPU: {gpu.name}")
            if gpu.memoryUtil < 0.5:
                available_gpus += gpus
            else:
                obs.log_warning(
                    "Cannot use this GPU because of its "
                    f"memory utilization @ {int(100.0 * gpu.memoryUtil)}%."
                )

        if len(available_gpus) > 1:
            obs.log_error(
                f"Only one GPU is supported right now, found {len(available_gpus)} available."
            )
            return False

        if len(available_gpus) < 1:
            obs.log_error("Need an available GPU!")
            return False

    except Exception as ex:
        obs.log_fatal(f"Could not read available VRAM: {str(ex)}")
        return False

    return True


def setup_tensorflow():
    # Try turning off tensorflow warnings
    # Sadly, this is broken again in 2.3 :(
    # https://github.com/tensorflow/tensorflow/issues/31870
    # To fix it, you need to call this method before importing anything
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    # Fix bug with tf 2.1.0 where cuDNN fails to initialize
    # See https://github.com/tensorflow/tensorflow/issues/24496
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    return True


class EnvironmentInfo:
    def __init__(self):
        import chia.version

        self.chia_version = chia.version.__version__
        self.environment_variables = dict(os.environ)
