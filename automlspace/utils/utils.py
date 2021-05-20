import os
import sys
import time
import contextlib
import traceback
import numpy as np
import pickle as pkl

seeds = [4465, 3822, 4531, 8459, 6295, 2854, 7820, 4050, 280, 6983,
         5497, 83, 9801, 8760, 5765, 6142, 4158, 9599, 1776, 1656]


# timer tool
@contextlib.contextmanager
def timeit(name=''):
    print("[%s]Start." % name, flush=True)
    start = time.time()
    yield
    end = time.time()
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    print("[%s]Total time = %d hours, %d minutes, %d seconds." % (name, h, m, s), flush=True)


def load_data(dataset, solnml_path):
    from solnml.datasets.utils import load_train_test_data
    train_node, test_node = load_train_test_data(dataset, data_dir=solnml_path, task_type=0)
    x_train, y_train = train_node.data
    x_val, y_val = test_node.data
    return x_train, y_train, x_val, y_val


def check_datasets(datasets, solnml_path):
    for _dataset in datasets:
        try:
            _ = load_data(_dataset, solnml_path)
        except Exception as e:
            print(traceback.format_exc())
            print('Dataset - %s load error: %s' % (_dataset, str(e)))
            raise


# ===== for plot =====

def descending(x):
    y = [x[0]]
    for i in range(1, len(x)):
        y.append(min(y[-1], x[i]))
    return y


def check_list(perf_list: list, max_run):
    if len(perf_list) > max_run:
        print('len of result: %d. max_run: %d. cut off.' % (len(perf_list), max_run))
        perf_list = perf_list[:max_run]
    if len(perf_list) < max_run:
        print('===== WARNING: len of result: %d. max_run: %d. extend.' % (len(perf_list), max_run))
        n = max_run - len(perf_list)
        perf_list.extend([perf_list[-1]] * n)
    return perf_list
