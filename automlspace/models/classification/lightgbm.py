from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter
from lightgbm import LGBMClassifier


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
    def get_hyperparameter_search_space(space_size='large'):
        """
            [('reg_alpha', 0.1570047322579923),
             ('learning_rate', 0.11772850320402356),
             ('colsample_bytree', 0.0293613412610731),
             ('n_estimators', 0.007886936947706083),
             ('min_child_weight', 0.007322490740197505),
             ('num_leaves', 0.0025417551872526653),
             ('reg_lambda', 0.0022570730772770295),
             ('subsample', 0.0006467968857933598),
             ('max_depth', 0.0)]
        """
        if space_size == 'large':
            cs = ConfigurationSpace()
            n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 1000, q=50, default_value=100)
            learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.9, log=True, default_value=0.1)
            num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128, q=10)

            reg_alpha = UniformFloatHyperparameter("reg_alpha", 0, 10, q=0.1, default_value=0)
            colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.1, 1, q=0.1, default_value=1)
            min_child_weight = UniformFloatHyperparameter("min_child_weight", 0, 10, q=0.1, default_value=1)

            reg_lambda = UniformFloatHyperparameter("reg_lambda", 1, 10, q=0.1, default_value=1)
            subsample = UniformFloatHyperparameter("subsample", 0.1, 1, q=0.1, default_value=1)
            max_depth = UnParametrizedHyperparameter('max_depth', 15)

            cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_weight, subsample,
                                    colsample_bytree, reg_alpha, reg_lambda])
        elif space_size == 'medium':
            cs = ConfigurationSpace()
            n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 1000, q=50, default_value=100)
            learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.9, log=True, default_value=0.1)
            num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128, q=10)

            reg_alpha = UniformFloatHyperparameter("reg_alpha", 0, 10, q=0.1, default_value=0)
            colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.1, 1, q=0.1, default_value=1)
            min_child_weight = UniformFloatHyperparameter("min_child_weight", 0, 10, q=0.1, default_value=1)

            reg_lambda = UnParametrizedHyperparameter("reg_lambda", 1)
            subsample = UnParametrizedHyperparameter("subsample", 1)
            max_depth = UnParametrizedHyperparameter('max_depth', 15)

            cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_weight, subsample,
                                    colsample_bytree, reg_alpha, reg_lambda])
        else:
            cs = ConfigurationSpace()
            n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 1000, q=50, default_value=100)
            learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.9, log=True, default_value=0.1)
            num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128, q=10)

            reg_alpha = UnParametrizedHyperparameter("reg_alpha", 0)
            colsample_bytree = UnParametrizedHyperparameter("colsample_bytree", 1)
            min_child_weight = UnParametrizedHyperparameter("min_child_weight", 1)

            reg_lambda = UnParametrizedHyperparameter("reg_lambda", 1)
            subsample = UnParametrizedHyperparameter("subsample", 1)
            max_depth = UnParametrizedHyperparameter('max_depth', 15)

            cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_weight, subsample,
                                    colsample_bytree, reg_alpha, reg_lambda])
        return cs
