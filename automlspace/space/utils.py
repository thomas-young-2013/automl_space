from typing import List
from ConfigSpace import Configuration, ConfigurationSpace


def sample_configurations(configuration_space: ConfigurationSpace,
                          historical_configs: List[Configuration],
                          sample_size: int):
    result = list()
    sample_cnt = 0
    if len(historical_configs) == 0:
        result.append(configuration_space.get_default_configuration())

    while len(result) < sample_size:
        config = configuration_space.sample_configuration(1)
        if config not in result and config not in historical_configs:
            result.append(config)
        sample_cnt += 1
        if sample_cnt > 50 * sample_size:
            result.append(config)
            break
    return result
