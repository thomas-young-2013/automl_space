import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter
from ConfigSpace import ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause
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


class LibLinear_SVC:
    # Liblinear is not deterministic as it uses a RNG inside
    def __init__(self, penalty, loss, dual, tol, C, multi_class,
                 fit_intercept, intercept_scaling, class_weight=None,
                 random_state=None):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.estimator = None
        self.time_limit = None

    def fit(self, X, Y):
        import sklearn.svm
        import sklearn.multiclass

        # In case of nested penalty
        if isinstance(self.penalty, dict):
            combination = self.penalty
            self.penalty = combination['penalty']
            self.loss = combination['loss']
            self.dual = combination['dual']

        self.C = float(self.C)
        self.tol = float(self.tol)

        self.dual = check_for_bool(self.dual)

        self.fit_intercept = check_for_bool(self.fit_intercept)

        self.intercept_scaling = float(self.intercept_scaling)

        if check_none(self.class_weight):
            self.class_weight = None

        estimator = sklearn.svm.LinearSVC(penalty=self.penalty,
                                          loss=self.loss,
                                          dual=self.dual,
                                          tol=self.tol,
                                          C=self.C,
                                          class_weight=self.class_weight,
                                          fit_intercept=self.fit_intercept,
                                          intercept_scaling=self.intercept_scaling,
                                          multi_class=self.multi_class,
                                          random_state=self.random_state)

        if len(Y.shape) == 2 and Y.shape[1] > 1:
            self.estimator = sklearn.multiclass.OneVsRestClassifier(estimator, n_jobs=1)
        else:
            self.estimator = estimator

        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_hyperparameter_search_space():
        cs = ConfigurationSpace()
        penalty = CategoricalHyperparameter(
            "penalty", ["l1", "l2"], default_value="l2")
        loss = CategoricalHyperparameter(
            "loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
        dual = CategoricalHyperparameter("dual", ['True', 'False'], default_value='True')
        # This is set ad-hoc
        tol = UniformFloatHyperparameter(
            "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
        C = UniformFloatHyperparameter(
            "C", 0.03125, 32768, log=True, default_value=1.0)
        multi_class = UnParametrizedHyperparameter("multi_class", "ovr")
        # These are set ad-hoc
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        intercept_scaling = UnParametrizedHyperparameter("intercept_scaling", 1)
        cs.add_hyperparameters([penalty, loss, dual, tol, C, multi_class,
                                fit_intercept, intercept_scaling])

        penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"),
            ForbiddenEqualsClause(loss, "hinge")
        )
        constant_penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(penalty, "l2"),
            ForbiddenEqualsClause(loss, "hinge")
        )
        penalty_and_dual = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "True"),
            ForbiddenEqualsClause(penalty, "l1")
        )
        cs.add_forbidden_clause(penalty_and_loss)
        cs.add_forbidden_clause(constant_penalty_and_loss)
        cs.add_forbidden_clause(penalty_and_dual)
        return cs


cs = LibLinear_SVC.get_hyperparameter_search_space()


def objective_func(config, x_train, x_val, y_train, y_val):
    conf_dict = config.get_dictionary()
    model = LibLinear_SVC(**conf_dict)
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

    algo_id = 'liblinear'

    run_exp(datasets, cs, rep_num, objective_func, algo_id, data_dir='../soln-ml/')
