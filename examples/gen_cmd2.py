"""
test ada-bo meta
2021.5.20
"""
#datasets = 'spambase,optdigits,satimage,wind,delta_ailerons,puma8NH,kin8nm,cpu_small,puma32H,cpu_act,bank32nh'
all_datasets = ['wind,delta_ailerons,puma8NH,kin8nm,cpu_small',
                'puma32H,cpu_act,bank32nh,mc1,delta_elevators']

cmd = 'nohup python examples/benchmark_automl_tuner.py --datasets %s --method %s --space_size large ' \
      '--algo %s --max_run 200 --n_jobs %d --meta_order %s --rep 10 >>%s 2>&1 &'

end = 'sleep 5\ntail -f %s\nsleep 1'

method_list = ['random-search', 'ada-bo', 'openbox', 'tpe']
algo_list = ['extra_trees', 'lightgbm', 'random_forest']    # 'adaboost'

n_jobs = 16

for method in method_list:
    if method == 'ada-bo':
        meta_list = ['yes', 'no']
    else:
        meta_list = ['no']
    for meta in meta_list:
        print('=====', method, meta)

        for algo in algo_list:
            log_file = None
            for idx, datasets in enumerate(all_datasets):
                log_file = f'data/nohup_data{idx+1}_200_{method}_{meta}_{algo}_large.log'
                cmd_line = cmd % (datasets, method, algo, n_jobs, meta, log_file)
                print(cmd_line)
            print(end % (log_file,))
            print()
        print()
