import os
import sys
import numpy as np
import pickle as pkl
from litebo.utils.config_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

sys.path.append(os.getcwd())
sys.path.append("../soln-ml")

from solnml.datasets.utils import load_train_test_data
from automlspace.adaptive_tuner import AdaptiveTuner

path = os.path.dirname(os.path.realpath(__file__))

train_node, test_node = load_train_test_data('spambase', data_dir='../soln-ml/', task_type=0)
x_train, y_train = train_node.data
x_val, y_val = test_node.data


class XGBoost:
    def __init__(self, n_estimators, learning_rate, max_depth, min_child_weight,
                 subsample, colsample_bytree, gamma=None, reg_alpha=None, reg_lambda=None,
                 n_jobs=4, seed=1):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.n_jobs = n_jobs
        self.random_state = np.random.RandomState(seed)
        self.estimator = None

    def fit(self, X, y):
        from xgboost import XGBClassifier
        # objective is set automatically in sklearn interface of xgboost
        self.estimator = XGBClassifier(
            use_label_encoder=False,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, q=10, default_value=500)
        max_depth = UniformIntegerHyperparameter("max_depth", 1, 12)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.9, log=True, default_value=0.1)
        min_child_weight = UniformFloatHyperparameter("min_child_weight", 0, 10, q=0.1, default_value=1)
        subsample = UniformFloatHyperparameter("subsample", 0.1, 1, q=0.1, default_value=1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.1, 1, q=0.1, default_value=1)
        gamma = UniformFloatHyperparameter("gamma", 0, 10, q=0.1, default_value=0)
        reg_alpha = UniformFloatHyperparameter("reg_alpha", 0, 10, q=0.1, default_value=0)
        reg_lambda = UniformFloatHyperparameter("reg_lambda", 1, 10, q=0.1, default_value=1)
        cs.add_hyperparameters([n_estimators, max_depth, learning_rate, min_child_weight, subsample,
                                colsample_bytree, gamma, reg_alpha, reg_lambda])
        return cs


cs = XGBoost.get_cs()


def objective_func(config):
    global x_train, x_val, y_train, y_val
    conf_dict = config.get_dictionary()
    model = XGBoost(**conf_dict, n_jobs=4, seed=1)
    model.fit(x_train, y_train)

    from sklearn.metrics import balanced_accuracy_score
    # evaluate on validation data
    y_pred = model.predict(x_val)
    perf = -balanced_accuracy_score(y_val, y_pred)  # minimize
    return perf


method = 'bo'
if method == 'random_search':
    rep_num = 1000
    X = []
    Y = []
    for i in range(rep_num):
        config = cs.sample_configuration()
        X.append(list(config.get_dictionary().values()))
        Y.append(objective_func(config))
        print('Iteration', i)
    X = np.array(X)
    Y = np.array(Y)
    with open('results.pkl', 'wb') as f:
        pkl.dump((X, Y), f)
elif method == 'bo':
    importance_list = ['n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree', 'gamma',
                       'min_child_weight',  'reg_alpha', 'reg_lambda', 'subsample']
    tuner = AdaptiveTuner(objective_func, cs, importance_list, max_run=100)
    tuner.run()
else:
    with open('results.pkl', 'rb') as f:
        X, Y = pkl.load(f)
