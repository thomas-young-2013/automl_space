from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter


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
    def get_hyperparameter_search_space(space_size='large'):
        """
            [('max_depth', 0.23882639703638706),
             ('learning_rate', 0.13215231139257277),
             ('n_estimators', 0.06237734379051598),
             ('algorithm', 0.0)]
        """
        cs = ConfigurationSpace()
        if space_size == 'large':
            max_depth = UniformIntegerHyperparameter(
                name="max_depth", lower=2, upper=8, default_value=3, log=False)
            learning_rate = UniformFloatHyperparameter(
                name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
            n_estimators = UniformIntegerHyperparameter(
                name="n_estimators", lower=50, upper=500, default_value=50, log=False)
            algorithm = CategoricalHyperparameter(
                name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
            cs.add_hyperparameters([n_estimators, learning_rate, algorithm, max_depth])
        elif space_size == 'medium':
            max_depth = UniformIntegerHyperparameter(
                name="max_depth", lower=2, upper=8, default_value=3, log=False)
            learning_rate = UniformFloatHyperparameter(
                name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
            n_estimators = UniformIntegerHyperparameter(
                name="n_estimators", lower=50, upper=500, default_value=50, log=False)
            algorithm = UnParametrizedHyperparameter("algorithm", "SAMME.R")
            cs.add_hyperparameters([n_estimators, learning_rate, algorithm, max_depth])
        else:
            max_depth = UniformIntegerHyperparameter(
                name="max_depth", lower=2, upper=8, default_value=3, log=False)
            learning_rate = UniformFloatHyperparameter(
                name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
            n_estimators = UnParametrizedHyperparameter("n_estimators", 50)
            algorithm = UnParametrizedHyperparameter("algorithm", "SAMME.R")
            cs.add_hyperparameters([n_estimators, learning_rate, algorithm, max_depth])
        return cs
