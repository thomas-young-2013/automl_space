"""
example cmdline:

python examples/benchmark_plot.py --datasets spambase,optdigits --algos xgboost --save_fig 0

"""
import argparse
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from utils import descending, check_list

default_datasets = 'spambase,optdigits,satimage,wind,delta_ailerons,puma8NH,kin8nm,cpu_small,puma32H,cpu_act,bank32nh'
default_algos = 'xgboost,lightgbm,adaboost,random_forest'
default_mths = 'random-search,lite-bo,tpe,ada-bo'
default_sizes = 'small,medium,large'

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default=default_datasets)
parser.add_argument('--algos', type=str, default=default_algos)
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--sizes', type=str, default=default_sizes)
parser.add_argument('--max_run', type=int, default=200)
parser.add_argument('--save_fig', type=int, default=0)

args = parser.parse_args()
datasets = args.datasets.split(',')
algos = args.algos.split(',')
mths = args.mths.split(',')
sizes = args.sizes.split(',')
max_run = args.max_run
save_fig = bool(args.save_fig)


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    color_list = ['purple', 'royalblue', 'green', 'brown', 'red', 'orange', 'yellowgreen', 'black', 'yellow']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x', 'd']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]

    for name in m_list:
        if name.startswith('random-search'):
            fill_values(name, 0)
        elif name.startswith('ada-bo'):
            fill_values(name, 4)
        elif name.startswith('lite-bo'):
            fill_values(name, 1)
        elif name.startswith('tpe'):
            fill_values(name, 2)
        else:
            print('color not defined:', name)
            fill_values(name, 8)
    return color_dict, marker_dict


def fetch_linestyle():
    linestyle_dict = {
        'small': ':',
        'medium': '--',
        'large': '-',
        '?': '-.',
    }
    return linestyle_dict


def get_mth_legend(mth):    # todo
    mth_lower = mth.lower()
    legend_dict = {
        'random': 'Random',
    }
    return legend_dict.get(mth_lower, mth)


def plot_setup(_dataset):   # todo
    if _dataset == 'covtype':
        plt.ylim(-0.937, -0.877)
        plt.xlim(0, max_run+5)
    elif _dataset.startswith('HIGGS'):
        plt.ylim(-0.7550, -0.7425)
        plt.xlim(0, max_run+5)


def plot(dataset, algo, save_fig=False):
    print('start', dataset)
    plot_setup(dataset)
    color_dict, marker_dict = fetch_color_marker(mths)
    linestyle_dict = fetch_linestyle()
    lw = 2
    markersize = 6
    markevery = int(max_run / 10)
    std_scale = 0.3
    alpha = 0.2

    min_val = 100000
    max_val = -100000

    for mth in mths:
        for space_size in sizes:
            stats = []
            dir_path = 'data/automl_tuner/%s-%d/%s-%s/%s/' % (dataset, max_run, algo, space_size, mth)
            if not os.path.exists(dir_path):
                print('===== ERROR: dir_path does not exist: %s' % (dir_path,))
                continue
            for file in os.listdir(dir_path):
                if file.startswith('record_%s-%s-%s-%s' % (mth, dataset, algo, space_size)) and file.endswith('.pkl'):
                    with open(os.path.join(dir_path, file), 'rb') as f:
                        save_item = pkl.load(f)
                    config_list, perf_list = save_item

                    perf_list = check_list(perf_list, max_run)
                    perf_list = descending(perf_list)
                    stats.append(perf_list)

            rep_num = len(stats)
            print('rep=%d  %s %s %s %s' % (rep_num, dataset, algo, mth, space_size))
            if rep_num == 0:
                print('===== ERROR: empty result')
                continue
            m = np.mean(stats, axis=0)
            s = np.std(stats, axis=0)
            x = np.arange(max_run) + 1

            # plot
            #label = get_mth_legend(mth)
            label = '%s-%s' % (mth, space_size)
            plt.plot(x, m, lw=lw, label=label, linestyle=linestyle_dict[space_size],
                     color=color_dict[mth],
                     #marker=marker_dict[mth], markersize=markersize, markevery=markevery,
                     )
            #plt.fill_between(x, m - s * std_scale, m + s * std_scale, alpha=alpha, facecolor=color_dict[mth])

            last_val = m[-1]
            min_val = min(min_val, last_val)
            max_val = max(max_val, last_val)

    # show plot
    gap = max_val - min_val
    plt.ylim(min_val - gap * 0.1, max_val + gap * 1.5)
    plt.xlim(0, max_run + 5)

    plt.legend(loc='upper right')
    plt.title('%s-%s' % (algo, dataset), fontsize=16)
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Negative validation score", fontsize=16)
    plt.tight_layout()
    if save_fig:
        dir_path = 'data/fig/'
        file_name = '%s-%s.jpg' % (algo, dataset)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(os.path.join(dir_path, file_name))
        plt.clf()
    else:
        plt.show()


for dataset in datasets:
    for algo in algos:
        plot(dataset, algo, save_fig=save_fig)

