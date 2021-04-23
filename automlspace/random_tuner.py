import numpy as np
from collections import OrderedDict
from .space.utils import sample_configurations


class RandomTuner(object):
    def __init__(self, objective_function, config_space, max_run=100, random_state=1):
        self.objective_function = objective_function
        self.config_space = config_space
        self.config_space.seed(random_state)
        self.max_run = max_run
        self.random_state = random_state
        self.history_dict = OrderedDict()

    def run(self):
        while len(self.history_dict.keys()) < self.max_run:
            self.iterate()

    def iterate(self):
        _config = sample_configurations(self.config_space, list(self.history_dict.keys()), 1)[0]
        _result = self.objective_function(_config)
        self.history_dict[_config] = _result

    def get_history(self):
        return list(self.history_dict.items())

    def get_incumbent(self):
        items = list(self.history_dict.items())
        inc_idx = np.argmin([_item[1] for _item in items])
        return items[inc_idx]
