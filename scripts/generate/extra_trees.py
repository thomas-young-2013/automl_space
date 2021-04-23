import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter
import argparse
import pickle as pkl
import os
import sys

sys.path.append(os.getcwd())

from scripts.utils import load_train_test_data, check_datasets, check_none, check_for_bool

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='None')
parser.add_argument('--rep_num', type=int, default=1000)


class ExtraTrees:

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split, max_features, bootstrap, max_leaf_nodes,
                 max_depth, min_weight_fraction_leaf, min_impurity_decrease,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 class_weight=None):

        self.n_estimators = n_estimators
        if criterion not in ("gini", "entropy"):
            raise ValueError("'criterion' is not in ('gini', 'entropy'): "
                             "%s" % criterion)
        self.criterion = criterion

        if check_none(max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(max_depth)
        if check_none(max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(max_leaf_nodes)

        self.min_samples_leaf = int(min_samples_leaf)
        self.min_samples_split = int(min_samples_split)
        self.max_features = float(max_features)
        self.bootstrap = check_for_bool(bootstrap)
        self.min_weight_fraction_leaf = float(min_weight_fraction_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.oob_score = oob_score
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        self.verbose = int(verbose)
        self.class_weight = class_weight
        self.estimator = None

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import ExtraTreesClassifier as ETC

        max_features = int(X.shape[1] ** float(self.max_features))
        self.estimator = ETC(n_estimators=self.n_estimators,
                             criterion=self.criterion,
                             max_depth=self.max_depth,
                             min_samples_split=self.min_samples_split,
                             min_samples_leaf=self.min_samples_leaf,
                             bootstrap=self.bootstrap,
                             max_features=max_features,
                             max_leaf_nodes=self.max_leaf_nodes,
                             min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                             min_impurity_decrease=self.min_impurity_decrease,
                             oob_score=self.oob_score,
                             n_jobs=self.n_jobs,
                             verbose=self.verbose,
                             random_state=self.random_state,
                             class_weight=self.class_weight,
                             warm_start=True)

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
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


cs = ExtraTrees.get_hyperparameter_search_space()


def objective_func(config, x_train, x_val, y_train, y_val):
    conf_dict = config.get_dictionary()
    model = ExtraTrees(**conf_dict, n_jobs=4)
    model.fit(x_train, y_train)

    from sklearn.metrics import balanced_accuracy_score
    # evaluate on validation data
    y_pred = model.predict(x_val)
    perf = -balanced_accuracy_score(y_val, y_pred)  # minimize
    return perf


if __name__ == '__main__':
    args = parser.parse_args()
    datasets = args.datasets.split(',')
    check_datasets(datasets, data_dir='../soln-ml/')
    rep_num = args.rep_num

    algo_id = 'extra_trees'

    for dataset in datasets:
        try:
            train_node, test_node = load_train_test_data(dataset, data_dir='../soln-ml/')
            train_x, train_y = train_node.data
            test_x, test_y = test_node.data

            X = []
            Y = []
            configs = cs.sample_configuration(rep_num)
            for config in configs:
                X.append(config.get_dictionary())
                Y.append(objective_func(config, train_x, test_x, train_y, test_y))
            with open('./data/%s_%s_%d.pkl' % (algo_id, dataset, rep_num), 'wb') as f:
                pkl.dump((X, Y), f)
        except Exception as e:
            print(e)
            print('%s failed!' % dataset)
