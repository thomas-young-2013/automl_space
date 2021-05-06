import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, Constant
from sklearn.preprocessing import OrdinalEncoder
import argparse
import pickle as pkl
import os
import sys

sys.path.insert(0, '.')
sys.path.append('../fanova')
from fanova import fANOVA

from scripts.generate.random_forest import RandomForest

parser = argparse.ArgumentParser()
parser.add_argument('--rep_num', type=int, default=500)

args = parser.parse_args()
rep_num = args.rep_num

cs = RandomForest.get_hyperparameter_search_space()
hp_order = []
hp_type = []
for hp in cs.get_hyperparameters():
    hp_order.append(hp.name)
    hp_type.append(hp)

individual_importance = {}
for key in hp_order:
    individual_importance[key] = []

for _, _, files in os.walk('./data'):
    for filename in files:
        try:
            if 'random_forest' in filename and '_%d' % rep_num in filename:
                with open(os.path.join('./data', filename), 'rb') as f:
                    config_list, perf_list = pkl.load(f)
                    perf_list = np.array(perf_list)
                    Y = 1 + perf_list

                    config_dict = {}
                    for hp_name in hp_order:
                        config_dict[hp_name] = []

                    for config in config_list:
                        for hp_name in hp_order:
                            config_dict[hp_name].append(config[hp_name])

                    for i, hp in enumerate(hp_type):
                        if isinstance(hp, CategoricalHyperparameter):
                            encoder = OrdinalEncoder()
                            config_dict[hp_order[i]] = encoder.fit_transform(
                                np.array(config_dict[hp_order[i]]).reshape(1, -1))[0]
                        elif isinstance(hp, UnParametrizedHyperparameter) or isinstance(hp, Constant):
                            config_dict[hp_order[i]] = [0] * len(config_dict[hp_order[i]])

                X = pd.DataFrame.from_dict(config_dict)

                f = fANOVA(X=X, Y=Y, config_space=cs)

                # marginal for first parameter
                for key in hp_order:
                    p_list = (key,)
                    importance = f.quantify_importance(p_list)[p_list]['total importance']
                    individual_importance[key].append(importance)
        except Exception as e:
            print(e)

print(individual_importance)
with open('random_forest_%d.pkl' % rep_num, 'wb') as f:
    pkl.dump(individual_importance, f)
