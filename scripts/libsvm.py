import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter
from ConfigSpace import EqualsCondition, InCondition
import argparse
import pickle as pkl
import os
import sys

sys.path.append(os.getcwd())

from scripts.utils import load_train_test_data, check_datasets, check_none, check_for_bool

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='None')
parser.add_argument('--rep_num', type=int, default=1000)


class LibSVM_SVC:
    def __init__(self, C, kernel, gamma, shrinking, tol, max_iter,
                 class_weight=None, degree=3, coef0=0, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y):
        import sklearn.svm
        # Nested kernel
        if isinstance(self.kernel, tuple):
            nested_kernel = self.kernel
            self.kernel = nested_kernel[0]
            if self.kernel == 'poly':
                self.degree = nested_kernel[1]['degree']
                self.coef0 = nested_kernel[1]['coef0']
            elif self.kernel == 'sigmoid':
                self.coef0 = nested_kernel[1]['coef0']

        self.C = float(self.C)
        if self.degree is None:
            self.degree = 3
        else:
            self.degree = int(self.degree)
        if self.gamma is None:
            self.gamma = 0.0
        else:
            self.gamma = float(self.gamma)
        if self.coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(self.coef0)
        self.tol = float(self.tol)
        self.max_iter = float(self.max_iter)

        self.shrinking = check_for_bool(self.shrinking)

        if check_none(self.class_weight):
            self.class_weight = None

        self.estimator = sklearn.svm.SVC(C=self.C,
                                         kernel=self.kernel,
                                         degree=self.degree,
                                         gamma=self.gamma,
                                         coef0=self.coef0,
                                         shrinking=self.shrinking,
                                         tol=self.tol,
                                         class_weight=self.class_weight,
                                         max_iter=self.max_iter,
                                         random_state=self.random_state,
                                         decision_function_shape='ovr')
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_hyperparameter_search_space():
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                                       default_value=1.0)
        # No linear kernel here, because we have liblinear
        kernel = CategoricalHyperparameter(name="kernel",
                                           choices=["rbf", "poly", "sigmoid"],
                                           default_value="rbf")
        degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)
        gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                           log=True, default_value=0.1)
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
        shrinking = CategoricalHyperparameter("shrinking", ["True", "False"],
                                              default_value="True")
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3,
                                         log=True)
        # cache size is not a hyperparameter, but an argument to the program!
        max_iter = UnParametrizedHyperparameter("max_iter", 10000)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking,
                                tol, max_iter])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)

        return cs


cs = LibSVM_SVC.get_hyperparameter_search_space()


def objective_func(config, x_train, x_val, y_train, y_val):
    conf_dict = config.get_dictionary()
    model = LibSVM_SVC(**conf_dict)
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

algo_id = 'libsvm'

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
        with open('%s_%s_%d.pkl' % (algo_id, dataset, rep_num), 'wb') as f:
            pkl.dump((X, Y), f)
    except Exception as e:
        print(e)
        print('%s failed!' % dataset)
