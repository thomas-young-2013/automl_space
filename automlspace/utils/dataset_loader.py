import os
import numpy as np
import pickle as pkl


def get_dataset_ids():
    data_file = os.path.join('data', 'info.pkl')
    with open(data_file, 'rb') as f:
        data = pkl.load(f)
    return list(data.keys())


def load_meta_data(algorithm='random_forest', dataset_ids=None, include_scaler=False):
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
    if include_scaler:
        return preproceess_data(result), scaler
    return preproceess_data(result)


def preproceess_data(data):
    hp_list = data[0]['hp_list']
    meta_features = [item['meta_feature'] for item in data]
    importance = [item['hp_importance'] for item in data]
    return np.asarray(meta_features), np.asarray(importance), hp_list


def load_meta_feature(dataset_id=''):
    data_file = os.path.join('data', 'cls_meta_dataset_embedding.pkl')
    with open(data_file, 'rb') as f:
        data = pkl.load(f)
    dataset_embeding = data['dataset_embedding']
    task_ids = data['task_ids']
    # print(dataset_embeding, task_ids)
    for _key, _value in zip(task_ids, dataset_embeding):
        if _key == 'init_' + dataset_id:
            return _value
    raise ValueError('Embeding for %s not found.' % dataset_id)
