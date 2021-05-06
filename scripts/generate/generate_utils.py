import traceback
import pickle as pkl
from scripts.utils import load_train_test_data, check_datasets


def run_exp(datasets, cs, rep_num, objective_func, algo_id, data_dir='../soln-ml/'):
    check_datasets(datasets, data_dir=data_dir)
    for dataset in datasets:
        try:
            train_node, test_node = load_train_test_data(dataset, data_dir=data_dir)
            train_x, train_y = train_node.data
            test_x, test_y = test_node.data

            X = []
            Y = []
            configs = cs.sample_configuration(rep_num * 1.2)    # prevent failure
            fail_num = 0
            for config in configs:
                try:
                    perf = objective_func(config, train_x, test_x, train_y, test_y)
                except Exception as e:
                    fail_num += 1
                    print(traceback.format_exc())
                    print('evaluation failed %d times!' % fail_num)
                else:
                    X.append(config.get_dictionary())
                    Y.append(perf)
                if len(X) >= rep_num:
                    break
            print('rep_num=%d. result_num=%d. fail_num=%d.' % (rep_num, len(X), fail_num))
            if len(X) > rep_num * 0.5:
                with open('./data/%s_%s_%d.pkl' % (algo_id, dataset, rep_num), 'wb') as f:
                    pkl.dump((X, Y), f)
            else:
                raise ValueError('too many failure. do not save.')
        except Exception as e:
            print(e)
            print('%s failed!' % dataset)
