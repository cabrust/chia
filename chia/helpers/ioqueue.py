import multiprocessing
import queue
import threading
import time
from typing import Any, Callable, Generator, Union

from chia import instrumentation


def make_generator_faster(
    gen: Callable[..., Generator[Any, None, None]],
    method: str,
    observable: instrumentation.Observable,
    max_buffer_size: int = 100,
):
    if method == "threading":
        return _make_generator_faster_threading(gen, max_buffer_size, observable)
    elif method == "multiprocessing":
        return _make_generator_faster_multiprocessing(gen, max_buffer_size, observable)
    elif method == "synchronous":
        return gen()
    else:
        raise ValueError(f"Unknown method {method}")


def _producer_main(
    item_queue: Union[queue.Queue, multiprocessing.Queue],
    gen: Callable[..., Generator[Any, None, None]],
    observable: instrumentation.Observable,
) -> None:
    gen_instance = gen()
    for item in gen_instance:
        item_queue.put(item)

    observable.log_debug("Producer done.")
    item_queue.put("THE_END")


def _consumer_generator(
    item_queue: Union[queue.Queue, multiprocessing.Queue],
    observable: instrumentation.Observable,
) -> Generator[Any, None, None]:

    waiting_begin = time.time()
    display_waiting_time_after_yield = False
    timeout_report_count = 0
    disable_timeout_message = False

    while True:
        try:
            item = item_queue.get(timeout=1.0)
            if item == "THE_END":
                observable.log_debug("Consumer done.")
                return
            else:
                yield item
                if display_waiting_time_after_yield:
                    # How long did we wait?
                    waiting_for = time.time() - waiting_begin
                    # Notify user that situation is resolved
                    observable.log_info(
                        f"Item retrieved with a waiting time of {waiting_for:.2f}s"
                    )
                    timeout_report_count += 1
                    if timeout_report_count >= 5:
                        observable.log_warning(
                            "Timeout encountered 5 times, will stop reporting now."
                        )
                        disable_timeout_message = True

                # Reset waiting timer and reporting
                waiting_begin = time.time()
                display_waiting_time_after_yield = False

        except queue.Empty:
            if (not display_waiting_time_after_yield) and (not disable_timeout_message):
                observable.log_info("Queue timeout hit, waiting for item...")
                display_waiting_time_after_yield = True
            time.sleep(1.0)


def _make_generator_faster_threading(
    gen: Callable[..., Generator[Any, None, None]],
    max_buffer_size: int,
    observable: instrumentation.Observable,
):
    item_queue = queue.Queue(maxsize=max_buffer_size)
    producer_thread = threading.Thread(
        target=_producer_main, args=(item_queue, gen, observable)
    )

    producer_thread.start()
    return _consumer_generator(item_queue, observable)


def _make_generator_faster_multiprocessing(
    gen: Callable[..., Generator[Any, None, None]],
    max_buffer_size: int,
    observable: instrumentation.Observable,
):
    item_queue = multiprocessing.Queue(maxsize=max_buffer_size)
    producer_process = multiprocessing.Process(
        target=_producer_main, args=(item_queue, gen, observable)
    )

    producer_process.start()
    return _consumer_generator(item_queue, observable)
