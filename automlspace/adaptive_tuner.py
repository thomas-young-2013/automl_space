from collections import OrderedDict
from litebo.optimizer.generic_smbo import SMBO
from litebo.utils.config_space import ConfigurationSpace
from ConfigSpace.util import deactivate_inactive_hyperparameters


class AdaptiveTuner(object):
    def __init__(self, objective_function, config_space, importance_list,
                 max_run=100, step_size=10, random_state=1):
        self.objective_function = objective_function
        self.importance_list = importance_list
        self.config_space = config_space
        self.step_size = step_size
        self.max_run = max_run
        self.random_state = random_state
        self._hp_cnt = 0
        self.history_dict = OrderedDict()

        # Obtain the default values.
        hps = self.config_space.get_hyperparameters()
        self.defaults = dict()
        for _hp in hps:
            self.defaults[_hp.name] = _hp.default_value

        # Safeness check.
        if len(importance_list) != len(self.config_space.get_hyperparameters()):
            raise ValueError('The length of importance list should be equal to the hyperspace\'s')
        hyper_names = [hp.name for hp in self.config_space.get_hyperparameters()]
        for _hp in importance_list:
            if _hp not in hyper_names:
                raise ValueError('HP name %s is not in the hyper-parameter space.' % _hp)

    def run(self):
        while len(self.history_dict.keys()) < self.max_run:
            self.iterate()

    def get_configspace(self):
        hp_num = self._hp_cnt + 1
        hps = self.config_space.get_hyperparameters()
        cs = ConfigurationSpace()
        for _id in range(hp_num):
            _hp_id = self.importance_list[_id]
            for _hp in hps:
                if _hp.name == _hp_id:
                    cs.add_hyperparameter(_hp)

        history_list = list()
        if len(self.history_dict.keys()) > 0 and hp_num < len(self.importance_list):
            new_hp = self.importance_list[self._hp_cnt]
            print('hp_num=', self._hp_cnt, 'new hp is', new_hp)
            for _config in self.history_dict.keys():
                # Impute the default value for new hyperparameter.
                _config_dict = _config.get_dictionary().copy()
                _config_dict[new_hp] = self.defaults[new_hp]
                history_list.append((_config_dict, self.history_dict[_config]))
        return cs, history_list

    def evaluate_wrapper(self, config):
        # Impute the missing hyper-parameters with default values.
        config_dict = config.get_dictionary()
        included_keys = config_dict.keys()
        all_keys = self.defaults.keys()
        for _missing_key in list(set(all_keys) - set(included_keys)):
            config_dict[_missing_key] = self.defaults[_missing_key]
        _config = deactivate_inactive_hyperparameters(configuration_space=self.config_space,
                                                      configuration=config_dict)
        return {'objs': (self.objective_function(_config),)}

    def iterate(self):
        config_space, hist_list = self.get_configspace()
        if len(hist_list) > 0:
            init_num = 0
        else:
            init_num = 3
        smbo = SMBO(self.evaluate_wrapper, config_space,
                    max_runs=self.step_size,
                    init_num=init_num, task_id='smbo%d' % self._hp_cnt)
        smbo.run()
        # Init the history trials.
        for _config_dict, _perf in hist_list:
            config = deactivate_inactive_hyperparameters(configuration_space=config_space,
                                                         configuration=_config_dict)
            smbo.config_advisor.configurations.append(config)
            smbo.config_advisor.perfs.append(_perf)

        # Save the runhistory.
        self.history_dict = OrderedDict()
        for _config, perf in zip(smbo.config_advisor.configurations, smbo.config_advisor.perfs):
            self.history_dict[_config] = perf

        if self._hp_cnt < len(self.importance_list) - 1:
            self._hp_cnt += 1