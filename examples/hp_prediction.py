import os
import sys
import numpy as np
import pickle as pkl
sys.path.append(os.getcwd())

"""
    data format:{
        dataset: {
            meta_features: {}, 
            adaboost: {},
            extra_trees: {},
            random_forest: {},
            lightgbm: {}
        }
    }
"""


def get_dataset_ids():
    data_file = os.path.join('data', 'info.pkl')
    with open(data_file, 'rb') as f:
        data = pkl.load(f)
    return list(data.keys())


def load_data(algorithm='random_forest', dataset_ids=None):
    data_file = os.path.join('data', 'info.pkl')
    with open(data_file, 'rb') as f:
        data = pkl.load(f)
    if dataset_ids is None:
        dataset_ids = data.keys()

    print('datasets', dataset_ids)
    meta_feature_ids = sorted(data[list(data.keys())[0]]['meta_features'].keys())
    hyperparameter_ids = sorted(data[list(data.keys())[0]][algorithm].keys())

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    meta_mat = list()
    for _key in data.keys():
        meta_vector = [data[_key]['meta_features'][_id] for _id in meta_feature_ids]
        meta_mat.append(meta_vector)
    scaler.fit(np.array(meta_mat), None)

    result = list()
    for _key in dataset_ids:
        meta_vector = [data[_key]['meta_features'][_id] for _id in meta_feature_ids]
        hp_importance = [data[_key][algorithm][_id] for _id in hyperparameter_ids]
        _item = dict()
        _item['meta_feature'] = scaler.transform(np.array([meta_vector]))[0]
        _item['hp_importance'] = hp_importance
        _item['hp_list'] = hyperparameter_ids.copy()
        result.append(_item)
    return preproceess_data(result)


def preproceess_data(data):
    hp_list = data[0]['hp_list']
    meta_features = [item['meta_feature'] for item in data]
    importance = [item['hp_importance'] for item in data]
    return np.asarray(meta_features), np.asarray(importance), hp_list


def average_precision_atN(preds, true_labels):
    N = len(preds)
    precision_ = list()
    for i in range(1, N+1):
        if preds[i-1] in true_labels[:i]:
            _pre = (len(precision_)+1)/i
            precision_.append(_pre)
    if len(precision_) == 0:
        return 0
    return np.sum(precision_) / N


def score(y1, y2):
    y1, y2 = -np.array(y1), -np.array(y2)
    y1_ids, y2_ids = np.argsort(y1), np.argsort(y2)
    return average_precision_atN(y2_ids, y1_ids), 1 if y1_ids[0] == y2_ids[0] else 0


def cross_validate(algorithm_id='random_forest'):
    from automlspace.ranknet import RankNetAdvisor
    dataset_ids = get_dataset_ids()
    np.random.shuffle(dataset_ids)
    n_fold = 10
    fold_size = len(dataset_ids) // n_fold
    aps, top1 = list(), list()

    for i in range(n_fold):
        test_datasets = dataset_ids[i * fold_size: (i + 1) * fold_size]
        advisor = RankNetAdvisor()

        train_datasets = [item for item in dataset_ids if item not in test_datasets]
        X_train, y_train, _ = load_data(algorithm=algorithm_id, dataset_ids=train_datasets)
        X_test, y_test, _ = load_data(algorithm=algorithm_id, dataset_ids=test_datasets)

        advisor.fit(X_train, y_train)

        for _x, _y in zip(X_test, y_test):
            y_pred = advisor.predict(_x)
            _ap, _top1 = score(_y, y_pred)
            aps.append(_ap)
            top1.append(_top1)

    print('Final AP@5', np.mean(aps), 'Top1', np.mean(top1))


def demo_evaluate(algorithm_id='random_forest'):
    X, y, labels = load_data(algorithm=algorithm_id, dataset_ids=None)
    from automlspace.ranknet import RankNetAdvisor
    advisor = RankNetAdvisor()
    advisor.fit(X, y)
    x = X[0]
    y_pred = advisor.predict(x)
    print(y_pred)
    print(y[0])
    print(score(y[0], y_pred))
    print(advisor.predict_ranking(x, rank_objs=labels))


if __name__ == "__main__":
    # demo_evaluate(algorithm_id='random_forest')
    cross_validate(algorithm_id='adaboost')
