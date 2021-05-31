import os
import sys
import numpy as np
import pickle as pkl
sys.path.append(os.getcwd())


from automlspace.utils.dataset_loader import get_dataset_ids, load_meta_data, preproceess_data

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
    n_fold = 5
    fold_size = len(dataset_ids) // n_fold
    aps, top1 = list(), list()

    for i in range(n_fold):
        test_datasets = dataset_ids[i * fold_size: (i + 1) * fold_size]
        advisor = RankNetAdvisor(algorithm_id=algorithm_id)

        train_datasets = [item for item in dataset_ids if item not in test_datasets]
        X_train, y_train, _ = load_meta_data(algorithm=algorithm_id, dataset_ids=train_datasets)
        X_test, y_test, _ = load_meta_data(algorithm=algorithm_id, dataset_ids=test_datasets)

        advisor.fit(X_train, y_train)

        for _x, _y in zip(X_test, y_test):
            y_pred = advisor.predict(_x)
            _ap, _top1 = score(_y, y_pred)
            aps.append(_ap)
            top1.append(_top1)

    print('Final AP@5', np.mean(aps), 'Top1', np.mean(top1))


def demo_evaluate(algorithm_id='random_forest'):
    X, y, labels = load_meta_data(algorithm=algorithm_id, dataset_ids=None)
    from automlspace.ranknet import RankNetAdvisor
    advisor = RankNetAdvisor(algorithm_id=algorithm_id)
    advisor.fit(X, y)
    x = X[0]
    y_pred = advisor.predict(x)
    print(y_pred)
    print(y[0])
    print(score(y[0], y_pred))
    print(advisor.predict_ranking(x, rank_objs=labels))


if __name__ == "__main__":
    """
    # 10-fold cv.
    random forest: 0.5775776014109348 Top1 0.7888888888888889
    lightgmb: 0.5550504605134234 Top1 0.4777777777777778
    adaboost: 0.7087962962962961 Top1 0.6777777777777778
    extra_trees: 0.5604907407407408 Top1 0.8222222222222222
    # 5-fold cv.
    random forest: 0.5583383458646617 Top1 0.7684210526315789
    lightgmb: 0.5246328784925276 Top1 0.47368421052631576
    adaboost: 0.7179824561403512 Top1 0.7263157894736842
    extra_trees: 0.564485380116959 Top1 0.8
    """
    # demo_evaluate(algorithm_id='random_forest')
    cross_validate(algorithm_id='extra_trees')
