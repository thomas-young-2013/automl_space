import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
sys.path.append("../soln-ml")

from solnml.components.feature_engineering.transformation_graph import DataNode
from solnml.utils.data_manager import DataManager


def load_train_test_data(dataset, data_dir='./', test_size=0.2, task_type=0, random_state=45):
    X, y, feature_type = load_data(dataset, data_dir)
    if task_type == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
    train_node = DataNode(data=[X_train, y_train], feature_type=feature_type.copy())
    test_node = DataNode(data=[X_test, y_test], feature_type=feature_type.copy())
    return train_node, test_node


def load_data(dataset, data_dir='./'):
    dm = DataManager()
    data_path = data_dir + 'data/openml100/%s.csv' % dataset

    # Load train data.
    if dataset in ['higgs', 'cjs', 'Australian', 'monks-problems-1', 'monks-problems-2', 'monks-problems-3', 'profb',
                   'JapaneseVowels']:
        label_column = 0
    else:
        label_column = -1

    header = 'infer'
    sep = ','

    train_data_node = dm.load_train_csv(data_path, label_col=label_column, header=header, sep=sep,
                                        na_values=["n/a", "na", "--", "-", "?"])
    train_data_node = dm.preprocess_fit(train_data_node)
    X, y = train_data_node.data
    feature_types = train_data_node.feature_types
    return X, y, feature_types


def check_datasets(datasets, data_dir='./'):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset, data_dir=data_dir)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


def check_true(p):
    if p in ("True", "true", 1, True):
        return True
    return False


def check_false(p):
    if p in ("False", "false", 0, False):
        return True
    return False


def check_none(p):
    if p in ("None", "none", None):
        return True
    return False


def check_for_bool(p):
    if check_false(p):
        return False
    elif check_true(p):
        return True
    else:
        raise ValueError("%s is not a bool" % str(p))
