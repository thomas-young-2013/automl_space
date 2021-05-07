
datasets = 'spambase,optdigits,satimage,wind,delta_ailerons,puma8NH,kin8nm,cpu_small,puma32H,cpu_act,bank32nh'


cmd = 'nohup python examples/benchmark_automl_tuner.py --datasets %s --method %s --space_size %s ' \
      '--algo %s --max_run 200 --step_size %d --n_jobs %d --rep 10 >>%s 2>&1 &'

end = 'sleep 5\ntail -f %s\nsleep 1'

method_list = ['random-search', 'ada-bo', 'lite-bo', 'tpe']
space_size_list = ['small', 'medium', 'large']
algo_list = ['xgboost', 'lightgbm', 'random_forest']    # 'adaboost'

step_size = 10
n_jobs = 16

for method in method_list:
    print('=====', method)
    for algo in algo_list:
        log_file = None
        for size in space_size_list:
            log_file = f'data/nohup_all_200_{method}_{algo}_{size}.log'
            cmd_line = cmd % (datasets, method, size, algo, step_size, n_jobs, log_file)
            print(cmd_line)
        print(end % (log_file,))
        print()
    print()
