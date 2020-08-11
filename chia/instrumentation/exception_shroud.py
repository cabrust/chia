import traceback

from chia import components
from chia.instrumentation import observable


class ExceptionShroud(observable.Observable):
    def __init__(self):
        observable.Observable.__init__(self)

    def __enter__(self):
        self.log_debug("Entering exception shroud")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.log_debug("Leaving exception shroud with exception")

            self.log_fatal(
                "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            )
            self.send_shutdown(successful=False)
        else:
            self.log_debug("Leaving exception shroud without exception")

        return False


class ExceptionShroudFactory(components.Factory):
    name_to_class_mapping = ExceptionShroud
