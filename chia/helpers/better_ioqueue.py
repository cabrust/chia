import queue
import threading
import time
from typing import Any, Callable, Generator

from chia import instrumentation

END_MARKER = "THIS_IS_THE_END"


def _fancy_job_producer(
    job_queue: queue.Queue,
    job_generator: Callable[..., Generator[Any, None, None]],
    end_count: int,
    observable: instrumentation.Observable,
) -> None:
    observable.log_debug("Job producer starting")

    # Create generator instance
    job_generator_instance = job_generator()

    for job in job_generator_instance:
        job_queue.put(job)

    for _ in range(end_count):
        job_queue.put(END_MARKER)

    observable.log_debug("Job producer done")


def _fancy_job_consumer(
    id: int,
    job_queue: queue.Queue,
    result_queue: queue.Queue,
    task,
    observable,
):
    observable.log_debug(f"Job consumer #{id + 1} starting")
    while True:
        job = job_queue.get()
        if job == END_MARKER:
            result_queue.put(END_MARKER)
            break
        else:
            result = task(job)
            result_queue.put(result)

    observable.log_debug(f"Job consumer #{id + 1} done")


def _fancy_result_consumer(
    result_queue: queue.Queue, end_count: int, observable, timeout=1.0
):
    observable.log_debug("Result consumer starting")
    observed_end_markers = 0

    state = "OK"
    first_attempt_at = None
    waiting_time_reports = 0

    while True:
        try:
            # Attempt to get result
            result = result_queue.get(timeout=timeout)

            # See if we should stop
            if result == END_MARKER:
                observed_end_markers += 1
                if observed_end_markers == end_count:
                    break
                else:
                    continue

            # Don't stop, we have an item
            if state == "TIMEOUT_HIT":
                waiting_time = time.time() - first_attempt_at
                observable.log_info(
                    f"Threaded queue item finally retrieved. Waiting time: {waiting_time:.1f}s"
                )
                waiting_time_reports += 1
                if waiting_time_reports == 5:
                    observable.log_warning(
                        "Threaded queue hit timeout five times, will not report anymore"
                    )
                    state = "GIVEN_UP"
                else:
                    state = "OK"

            yield result

        except queue.Empty:
            if state == "OK":
                state = "TIMEOUT_HIT"
                observable.log_info("Threaded queue timeout hit")
                first_attempt_at = time.time() - timeout

    observable.log_debug("Result consumer done")


def threaded_processor(
    job_generator,
    task,
    observable,
    max_job_queue_size=4096,
    max_result_queue_size=64,
    num_threads=8,
):
    job_queue = queue.Queue(maxsize=max_job_queue_size)
    result_queue = queue.Queue(maxsize=max_result_queue_size)

    # Start job producer thread
    job_producer_thread = threading.Thread(
        target=_fancy_job_producer,
        args=(job_queue, job_generator, num_threads, observable),
    )

    job_producer_thread.start()

    # Start job consumer threads
    job_consumer_threads = []
    for i in range(num_threads):
        job_consumer_thread = threading.Thread(
            target=_fancy_job_consumer,
            args=(i, job_queue, result_queue, task, observable),
        )
        job_consumer_thread.start()
        job_consumer_threads += [job_producer_thread]

    # Consume and output results
    return _fancy_result_consumer(result_queue, num_threads, observable)
