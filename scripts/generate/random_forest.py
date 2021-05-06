import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter
import argparse
import pickle as pkl
import os
import sys

sys.path.insert(0, '.')
from generate_utils import run_exp
from scripts.utils import check_none, check_for_bool

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='None')
parser.add_argument('--rep_num', type=int, default=1000)


class RandomForest:
    def __init__(self, n_estimators, criterion, max_features,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, bootstrap, max_leaf_nodes,
                 min_impurity_decrease, random_state=None, n_jobs=1,
                 class_weight=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.estimator = None

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import RandomForestClassifier
        self.n_estimators = int(self.n_estimators)
        self.max_depth = int(self.max_depth)

        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

        if self.max_features not in ("sqrt", "log2", "auto"):
            max_features = int(X.shape[1] ** float(self.max_features))
        else:
            max_features = self.max_features

        self.bootstrap = check_for_bool(self.bootstrap)

        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)

        self.min_impurity_decrease = float(self.min_impurity_decrease)

        # initial fit of only increment trees
        self.estimator = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=max_features,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            bootstrap=self.bootstrap,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
            warm_start=True)

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter('n_estimators', 50, 1000, default_value=100)
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default_value="gini")
        max_features = UniformFloatHyperparameter(
            "max_features", 0., 1., default_value=0.5)

        max_depth = UniformIntegerHyperparameter("max_depth", 4, 12, default_value=5)
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True")
        cs.add_hyperparameters([n_estimators, criterion, max_features,
                                max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                bootstrap, min_impurity_decrease])
        return cs


cs = RandomForest.get_hyperparameter_search_space()


def objective_func(config, x_train, x_val, y_train, y_val):
    conf_dict = config.get_dictionary()
    model = RandomForest(**conf_dict, n_jobs=4)
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

    algo_id = 'random_forest'

    run_exp(datasets, cs, rep_num, objective_func, algo_id, data_dir='../soln-ml/')
