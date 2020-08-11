import abc
import math

from chia import components


class KerasLearningRateSchedule(abc.ABC):
    @abc.abstractmethod
    def __call__(self, step):
        pass


class ConstantKerasLearningRateSchedule(KerasLearningRateSchedule):
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr

    def __call__(self, step):
        return self.initial_lr


class SGDRKerasLearningRateSchedule(KerasLearningRateSchedule):
    def __init__(
        self,
        minimum_lr: float,
        maximum_lr: float,
        T_0: int,
        T_mult: int = 2,
        maximum_lr_decay: float = 1.0,
        warmup_steps: int = 0,
        warmup_lr: float = 0.1,
    ):
        self.minimum_lr = minimum_lr
        self.maximum_lr = maximum_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.maximum_lr_decay = maximum_lr_decay
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.warmup_lr

        step_cur = step
        maximum_lr_cur = self.maximum_lr
        T_i = self.T_0
        while step_cur > T_i:
            step_cur -= T_i
            T_i *= self.T_mult
            maximum_lr_cur *= self.maximum_lr_decay

        T_cur = float(step_cur)
        T_i = float(T_i)

        maximum_lr_cur = max(maximum_lr_cur, self.minimum_lr)

        eta_t = self.minimum_lr + 0.5 * (maximum_lr_cur - self.minimum_lr) * (
            1.0 + math.cos(math.pi * (T_cur / T_i))
        )
        return eta_t


class ExponentialKerasLearningRateSchedule(KerasLearningRateSchedule):
    def __init__(self, initial_lr, end_lr, decay_factor, steps_per_decay):
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.decay_factor = decay_factor
        self.steps_per_decay = steps_per_decay

    def __call__(self, step):
        current_decay_exponent = float(step) / self.steps_per_decay
        current_decay_factor = math.pow(self.decay_factor, current_decay_exponent)
        current_lr = max(self.initial_lr * current_decay_factor, self.end_lr)
        return current_lr


class KerasLearningRateScheduleFactory(components.Factory):
    name_to_class_mapping = {
        "constant": ConstantKerasLearningRateSchedule,
        "sgdr": SGDRKerasLearningRateSchedule,
        "exponential": ExponentialKerasLearningRateSchedule,
    }
    default_section = "keras_learning_rate_schedule"
