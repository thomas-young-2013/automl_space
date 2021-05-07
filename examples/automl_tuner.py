import os
import sys
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from litebo.optimizer.generic_smbo import SMBO

sys.path.insert(0, '.')
sys.path.append("../soln-ml")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='spambase')
parser.add_argument('--method', type=str, default='ada-bo')
parser.add_argument('--space_size', type=str, default='large')
parser.add_argument('--algo', type=str, default='xgboost')
parser.add_argument('--max_run', type=int, default=30)

args = parser.parse_args()
max_run = args.max_run
method = args.method
algo = args.algo

from solnml.datasets.utils import load_train_test_data
from automlspace.random_tuner import RandomTuner
from automlspace.adaptive_tuner import AdaptiveTuner
from automlspace.models.classification.lightgbm import LightGBM
from automlspace.models.classification.xgboost import XGBoost
from automlspace.models.classification.adaboost import Adaboost
from automlspace.models.classification.random_forest import RandomForest

path = os.path.dirname(os.path.realpath(__file__))

train_node, test_node = load_train_test_data(args.dataset, data_dir='../soln-ml/', task_type=0)
x_train, y_train = train_node.data
x_val, y_val = test_node.data

if algo == 'xgboost':
    model_class = XGBoost
elif algo == 'lightgbm':
    model_class = LightGBM
elif algo == 'adaboost':
    model_class = Adaboost
elif algo == 'random_forest':
    model_class = RandomForest
else:
    raise ValueError('Invalid algorithm~')
cs = model_class.get_hyperparameter_search_space(space_size=args.space_size)
print(cs)

def objective_func(config):
    global x_train, x_val, y_train, y_val
    conf_dict = config.get_dictionary()
    if algo == 'xgboost':
        model = XGBoost(**conf_dict, n_jobs=4, seed=1)
    elif algo == 'lightgbm':
        model = LightGBM(**conf_dict)
    elif algo == 'adaboost':
        model = Adaboost(**conf_dict)
    elif algo == 'random_forest':
        model = RandomForest(**conf_dict)
    else:
        raise ValueError('Invalid algorithm~')

    model.fit(x_train, y_train)

    from sklearn.metrics import balanced_accuracy_score
    # evaluate on validation data
    y_pred = model.predict(x_val)
    perf = -balanced_accuracy_score(y_val, y_pred)  # minimize
    return perf


if method == 'random-search':
    tuner = RandomTuner(objective_func, cs, max_run=max_run)
    tuner.run()
    print(tuner.get_incumbent())
elif method == 'ada-bo':
    if algo == 'xgboost':
        importance_list = ['n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree', 'gamma',
                           'min_child_weight', 'reg_alpha', 'reg_lambda', 'subsample']
    elif algo == 'lightgbm':
        importance_list = ['n_estimators', 'learning_rate', 'num_leaves', 'reg_alpha', 'colsample_bytree',
                           'min_child_weight', 'reg_lambda', 'subsample', 'max_depth']
    elif algo == 'adaboost':
        importance_list = ['n_estimators', 'learning_rate', 'max_depth', 'algorithm']
    elif algo == 'random_forest':
        importance_list = ['n_estimators', 'max_depth', 'max_features', 'min_samples_leaf',
                           'min_samples_split', 'bootstrap', 'criterion', 'max_leaf_nodes',
                           'min_impurity_decrease', 'min_weight_fraction_leaf']
    else:
        raise ValueError('Invalid algorithm~')

    tuner = AdaptiveTuner(objective_func, cs, importance_list, max_run=max_run, step_size=10)
    tuner.run()
    print(tuner.get_incumbent())
elif method == 'lite-bo':
    bo = SMBO(objective_func, cs,
              advisor_type='default',
              max_runs=max_run,
              task_id='tuning-litebo',
              logging_dir='logs')
    bo.run()
    print(bo.get_incumbent())
elif method == 'tpe':
    bo = SMBO(objective_func, cs,
              advisor_type='default',
              max_runs=max_run,
              task_id='tuning-tpe',
              logging_dir='logs')
    bo.run()
    print(bo.get_incumbent())
else:
    raise ValueError('Invalid method id - %s.' % args.method)


# TODO: 1) save result, 2) plot convergence
