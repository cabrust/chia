import time

from PIL import Image

from chia import instrumentation


class NetworkResistantImage:
    @staticmethod
    def open(filename: str, observable: instrumentation.Observable):
        cumulative_wait_max = 7200.0
        wait_interval_initial = 0.5

        # Set initial state
        wait_interval = wait_interval_initial
        cumulative_wait = 0.0
        last_exception = None

        while cumulative_wait <= cumulative_wait_max:
            try:
                image = Image.open(filename)
                return image

            except Exception as ex:
                observable.log_warning(
                    f"Cannot open {filename}. Waiting {wait_interval} seconds."
                )
                last_exception = ex
                time.sleep(wait_interval)
                cumulative_wait += wait_interval
                wait_interval *= 2.0

        raise last_exception
