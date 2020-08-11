import abc

from chia import instrumentation


class Runner(abc.ABC, instrumentation.Observable):
    def __init__(self, experiment_container):
        instrumentation.Observable.__init__(self)
        self.experiment_container = experiment_container

    @abc.abstractmethod
    def run(self):
        pass
