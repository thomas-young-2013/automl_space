import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter
import argparse
import pickle as pkl
import os
import sys
from lightgbm import LGBMClassifier

sys.path.insert(0, '.')
from scripts.generate.generate_utils import run_exp
from scripts.utils import check_none, check_for_bool

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='None')
parser.add_argument('--rep_num', type=int, default=1000)


class LightGBM:
    def __init__(self, n_estimators, num_leaves, learning_rate, max_depth, min_child_weight,
                 subsample, colsample_bytree, reg_alpha, reg_lambda, n_jobs=4, random_state=None):
        self.n_estimators = int(n_estimators)
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        self.estimator = LGBMClassifier(n_estimators=self.n_estimators,
                                        num_leaves=self.num_leaves,
                                        max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        min_child_weight=self.min_child_weight,
                                        subsample=self.subsample,
                                        colsample_bytree=self.colsample_bytree,
                                        reg_alpha=self.reg_alpha,
                                        reg_lambda=self.reg_lambda,
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, q=50, default_value=500)
        num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
        max_depth = UnParametrizedHyperparameter('max_depth', 15)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.9, log=True, default_value=0.1)
        min_child_weight = UniformFloatHyperparameter("min_child_weight", 0, 10, q=0.1, default_value=1)
        subsample = UniformFloatHyperparameter("subsample", 0.1, 1, q=0.1, default_value=1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.1, 1, q=0.1, default_value=1)
        reg_alpha = UniformFloatHyperparameter("reg_alpha", 0, 10, q=0.1, default_value=0)
        reg_lambda = UniformFloatHyperparameter("reg_lambda", 1, 10, q=0.1, default_value=1)
        cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_weight, subsample,
                                colsample_bytree, reg_alpha, reg_lambda])
        return cs


cs = LightGBM.get_hyperparameter_search_space()


def objective_func(config, x_train, x_val, y_train, y_val):
    conf_dict = config.get_dictionary()
    model = LightGBM(**conf_dict)
    model.fit(x_train, y_train)

    from sklearn.metrics import balanced_accuracy_score
    # evaluate on validation data
    y_pred = model.predict(x_val)
    perf = -balanced_accuracy_score(y_val, y_pred)  # minimize
    return perf


if __name__ == '__main__':
    args = parser.parse_args()
    datasets = args.datasets.split(',')
    rep_num = args.rep_num

    algo_id = 'lightgbm'

    run_exp(datasets, cs, rep_num, objective_func, algo_id, data_dir='../soln-ml/')
