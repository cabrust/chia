from chia import components
from chia.instrumentation.observers import json_observer, stream_observer


class ObserverFactory(components.Factory):
    name_to_class_mapping = {
        "stream": stream_observer.StreamObserver,
        "json": json_observer.JSONObserver,
    }
