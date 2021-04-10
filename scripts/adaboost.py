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


class Adaboost:
    def __init__(self, n_estimators, learning_rate, algorithm, max_depth,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        self.max_depth = max_depth
        self.estimator = None

    def fit(self, X, Y, sample_weight=None):
        import sklearn.tree
        import sklearn.ensemble

        self.n_estimators = int(self.n_estimators)
        self.learning_rate = float(self.learning_rate)
        self.max_depth = int(self.max_depth)
        base_estimator = sklearn.tree.DecisionTreeClassifier(max_depth=self.max_depth)

        estimator = sklearn.ensemble.AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            random_state=self.random_state
        )

        estimator.fit(X, Y, sample_weight=sample_weight)

        self.estimator = estimator
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()

        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=50, log=False)
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
        algorithm = CategoricalHyperparameter(
            name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=2, upper=8, default_value=3, log=False)

        cs.add_hyperparameters([n_estimators, learning_rate, algorithm, max_depth])
        return cs


cs = Adaboost.get_hyperparameter_search_space()


def objective_func(config, x_train, x_val, y_train, y_val):
    conf_dict = config.get_dictionary()
    model = Adaboost(**conf_dict)
    model.fit(x_train, y_train)

    from sklearn.metrics import balanced_accuracy_score
    # evaluate on validation data
    y_pred = model.predict(x_val)
    perf = -balanced_accuracy_score(y_val, y_pred)  # minimize
    return perf


args = parser.parse_args()
datasets = args.datasets.split(',')
check_datasets(datasets, data_dir='../soln-ml/')
rep_num = args.rep_num

algo_id = 'adaboost'

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
