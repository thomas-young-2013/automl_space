"""
python examples/benchmark_automl_tuner.py --datasets spambase --method ada-bo --space_size large --algo xgboost --max_run 200 --step_size 10 --rep 1 --start_id 0

"""

import os
import sys
import time
import argparse
import traceback
import pickle as pkl

openbox_path = '../open-box/'
solnml_path = '../soln-ml/'

sys.path.insert(0, '.')
sys.path.insert(1, openbox_path)
sys.path.insert(2, solnml_path)


from automlspace.random_tuner import RandomTuner
from automlspace.adaptive_tuner import AdaptiveTuner
from automlspace.models.classification.lightgbm import LightGBM
from automlspace.models.classification.xgboost import XGBoost
from automlspace.models.classification.adaboost import Adaboost
from automlspace.models.classification.random_forest import RandomForest
from automlspace.models.classification.extra_trees import ExtraTrees

from automlspace.utils.utils import seeds, timeit, load_data, check_datasets
from automlspace.utils.dataset_loader import load_meta_data, load_meta_feature

from keras import backend as K  # for segmentation fault bug

path = os.path.dirname(os.path.realpath(__file__))

default_datasets = 'spambase,optdigits,satimage,wind,delta_ailerons,puma8NH,kin8nm,cpu_small,puma32H,cpu_act,bank32nh'

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default=default_datasets)
parser.add_argument('--method', type=str, default='ada-bo')  # random-search, openbox, tpe, ada-bo
parser.add_argument('--space_size', type=str, default='large', choices=['large', 'medium', 'small'])
parser.add_argument('--algo', type=str, default='xgboost')  # xgboost, lightgbm, adaboost, random_forest
parser.add_argument('--strategy', type=str, default='default')  # default, tpe, random. for AdaptiveTuner
parser.add_argument('--max_run', type=int, default=200)
parser.add_argument('--step_size', type=int, default=10)  # for AdaptiveTuner
parser.add_argument('--n_jobs', type=int, default=4)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--meta_order', type=str, default='no', choices=['no', 'yes'])


args = parser.parse_args()
datasets = args.datasets.split(',')
method = args.method
space_size = args.space_size
algo = args.algo
max_run = args.max_run
step_size = args.step_size
strategy = args.strategy
use_meta_order = args.meta_order

n_jobs = args.n_jobs
rep = args.rep
start_id = args.start_id


def evaluate(dataset, method, algo, space_size, max_run, step_size, seed):
    if algo == 'xgboost':
        model_class = XGBoost
    elif algo == 'lightgbm':
        model_class = LightGBM
    elif algo == 'adaboost':
        model_class = Adaboost
    elif algo == 'random_forest':
        model_class = RandomForest
    elif algo == 'extra_trees':
        model_class = ExtraTrees
    else:
        raise ValueError('Invalid algorithm: %s!' % algo)
    cs = model_class.get_hyperparameter_search_space(space_size=space_size)

    x_train, y_train, x_val, y_val = load_data(dataset, solnml_path)

    def objective_func(config):
        conf_dict = config.get_dictionary()
        if algo == 'xgboost':
            model = XGBoost(**conf_dict, n_jobs=n_jobs, seed=1)
        elif algo == 'lightgbm':
            model = LightGBM(**conf_dict, n_jobs=n_jobs, random_state=1)
        elif algo == 'adaboost':
            model = Adaboost(**conf_dict, random_state=1)
        elif algo == 'random_forest':
            model = RandomForest(**conf_dict, n_jobs=n_jobs, random_state=1)
        elif algo == 'extra_trees':
            model = ExtraTrees(**conf_dict, n_jobs=n_jobs, random_state=1)
        else:
            raise ValueError('Invalid algorithm: %s' % algo)

        model.fit(x_train, y_train)

        from sklearn.metrics import balanced_accuracy_score
        # evaluate on validation data
        y_pred = model.predict(x_val)
        perf = -balanced_accuracy_score(y_val, y_pred)  # minimize
        return perf

    if method == 'random-search':
        # tuner = RandomTuner(objective_func, cs, max_run=max_run, random_state=seed)
        # tuner.run()
        # print(tuner.get_incumbent())
        # config_list = list(tuner.history_dict.keys())
        # perf_list = list(tuner.history_dict.values())
        from openbox.optimizer.generic_smbo import SMBO
        task_id = 'tuning-random-%s-%s-%s-%d' % (dataset, algo, space_size, seed)
        bo = SMBO(objective_func, cs,
                  advisor_type='random',
                  max_runs=max_run,
                  task_id=task_id,
                  logging_dir='logs',
                  random_state=seed)
        bo.run()
        print(bo.get_incumbent())
        history = bo.get_history()
        config_list = history.configurations
        perf_list = history.perfs
    elif method == 'ada-bo':
        if algo == 'xgboost':
            importance_list = ['n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree', 'gamma',
                               'min_child_weight', 'reg_alpha', 'reg_lambda', 'subsample']
        elif algo == 'lightgbm':
            importance_list = ['n_estimators', 'learning_rate', 'num_leaves', 'reg_alpha', 'colsample_bytree',
                               'min_child_weight', 'reg_lambda', 'subsample', 'max_depth']
        elif algo == 'adaboost':
            importance_list = ['n_estimators', 'learning_rate', 'max_depth', 'algorithm']
        elif algo == 'random_forest':
            importance_list = ['n_estimators', 'max_depth', 'max_features', 'min_samples_leaf',
                               'min_samples_split', 'bootstrap', 'criterion', 'max_leaf_nodes',
                               'min_impurity_decrease', 'min_weight_fraction_leaf']
        elif algo == 'extra_trees':
            importance_list = ['n_estimators', 'max_depth', 'max_features', 'min_samples_leaf',
                               'min_samples_split', 'bootstrap', 'criterion', 'max_leaf_nodes',
                               'min_impurity_decrease', 'min_weight_fraction_leaf']
        else:
            raise ValueError('Invalid algorithm~')
        print('Previous important list is', ','.join(importance_list))

        if use_meta_order == "yes":
            data_, scaler_ = load_meta_data(algorithm=algo, dataset_ids=None, include_scaler=True)
            X, y, labels = data_

            from automlspace.ranknet import RankNetAdvisor
            advisor = RankNetAdvisor(algorithm_id=algo)
            advisor.fit(X, y)

            new_embeding = load_meta_feature(dataset_id=dataset)
            new_embeding = scaler_.transform([new_embeding])[0]
            importance_list = advisor.predict_ranking(new_embeding, rank_objs=labels)
            print('New important list is', ','.join(importance_list))

        tuner = AdaptiveTuner(objective_func, cs, importance_list,
                              strategy=strategy, max_run=max_run, step_size=step_size, random_state=seed)
        tuner.run()
        print(tuner.get_incumbent())
        config_list = list(tuner.history_dict.keys())
        perf_list = list(tuner.history_dict.values())
    elif method == 'openbox':
        from openbox.optimizer.generic_smbo import SMBO
        task_id = 'tuning-openbox-%s-%s-%s-%d' % (dataset, algo, space_size, seed)
        bo = SMBO(objective_func, cs,
                  advisor_type='default',
                  max_runs=max_run,
                  task_id=task_id,
                  logging_dir='logs',
                  random_state=seed)
        bo.run()
        print(bo.get_incumbent())
        history = bo.get_history()
        config_list = history.configurations
        perf_list = history.perfs
    elif method == 'tpe':
        from openbox.optimizer.generic_smbo import SMBO
        task_id = 'tuning-tpe-%s-%s-%s-%d' % (dataset, algo, space_size, seed)
        bo = SMBO(objective_func, cs,
                  advisor_type='tpe',
                  max_runs=max_run,
                  task_id=task_id,
                  logging_dir='logs',
                  random_state=seed)
        bo.run()
        print(bo.get_incumbent())
        history = bo.get_history()
        config_list = history.configurations
        perf_list = history.perfs
    else:
        raise ValueError('Invalid method id - %s.' % args.method)

    if len(config_list) > max_run:
        print('len of result: %d. max_run: %d. cut off.' % (len(config_list), max_run))
        config_list = config_list[:max_run]
        perf_list = perf_list[:max_run]
    if len(config_list) < max_run:
        print('===== WARNING: len of result: %d. max_run: %d.' % (len(config_list), max_run))
    return config_list, perf_list


check_datasets(datasets, solnml_path)
with timeit('all'):
    for dataset in datasets:
        for i in range(start_id, start_id + rep):
            seed = seeds[i]
            with timeit('%s %s %s %s-%d-%d' % (method, algo, space_size, dataset, i, seed)):
                try:
                    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                    method_str = method
                    if method == 'ada-bo':
                        if strategy != 'default':
                            method_str += '-%s' % strategy
                        if use_meta_order == 'yes':
                            method_str += ''
                    method_id = method_str + '-%s-%s-%s-%04d-%s' % (dataset, algo, space_size, seed, timestamp)

                    config_list, perf_list = evaluate(dataset, method, algo, space_size, max_run, step_size, seed)

                    save_item = (config_list, perf_list)
                    dir_path = 'data/automl_tuner/%s-%d/%s-%s/%s/' % (dataset, max_run, algo, space_size, method_str)
                    file_name = 'record_%s.pkl' % (method_id,)
                    try:
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                    except FileExistsError:
                        pass
                    with open(os.path.join(dir_path, file_name), 'wb') as f:
                        pkl.dump(save_item, f)
                    print(dir_path, file_name, 'saved!', flush=True)
                except Exception as e:
                    print(traceback.format_exc())
                    print(dataset, i, seed, method, algo, 'error!')
