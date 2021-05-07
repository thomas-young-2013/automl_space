
datasets = 'spambase,optdigits,satimage,wind,delta_ailerons,puma8NH,kin8nm,cpu_small,puma32H,cpu_act,bank32nh'
datasets = datasets.split(',')

space_size_list = ['small', 'medium', 'large']
algo_list = ['xgboost', 'lightgbm', 'random_forest']    # 'adaboost'
max_run = 200

header = """#!/bin/bash
if [ $# != 1 ];then
    echo -n "Enter method: "; read method
else
    method=$1
fi
echo method is $method
cd `dirname $0`
cd automl_tuner
"""
print(header)
for dataset in datasets:
    for algo in algo_list:
        for size in space_size_list:
            dir_path = '%s-%d/%s-%s/$method' % (dataset, max_run, algo, size)
            cmd = 'echo %s; ls %s | wc -l' % (dir_path, dir_path)
            print(cmd)
