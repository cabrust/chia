import os


def setup_environment():
    enable_segfault_catching()
    validate_gpu_configuration()
    setup_tensorflow()


def enable_segfault_catching():
    import faulthandler

    faulthandler.enable()
    return True


def validate_gpu_configuration():
    try:
        import os

        from GPUtil import GPUtil

        # Disable GPUS if desired
        if "CHIA_CPU_ONLY" in os.environ.keys():
            if os.environ["CHIA_CPU_ONLY"] == "1":
                print(
                    "Requested CPU only operation."
                    "Disabling all GPUS via environment variable."
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                return True

        gpus = GPUtil.getGPUs()
        if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
            cuda_filter = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            print(f"Found CUDA device filter: {cuda_filter}")
            gpus = [gpu for gpu in gpus if str(gpu.id) in cuda_filter]

        available_gpus = []
        for gpu in gpus:
            print(f"Found GPU: {gpu.name}")
            if gpu.memoryUtil < 0.5:
                available_gpus += gpus
            else:
                print(
                    "Cannot use this GPU because of its "
                    f"memory utilization @ {int(100.0 * gpu.memoryUtil)}%."
                )

        if len(available_gpus) > 1:
            print(
                f"Only one GPU is supported right now, found {len(available_gpus)} available."
            )
            return False

        if len(available_gpus) < 1:
            print("Need an available GPU!")
            return False

    except Exception:
        print("Could not read available VRAM.")

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
        import chia

        self.chia_version = chia.__version__
        self.environment_variables = dict(os.environ)
