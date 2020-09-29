from chia import components
from chia.instrumentation.observers import (
    buffered_observer,
    json_observer,
    stream_observer,
)


class ObserverFactory(components.Factory):
    name_to_class_mapping = {
        "buffered": buffered_observer.BufferedObserver,
        "stream": stream_observer.StreamObserver,
        "json": json_observer.JSONObserver,
    }
