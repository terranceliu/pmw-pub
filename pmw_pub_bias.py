import sys
sys.path.append("../private-pgm/src")

from mwem import *
from Util.data_pub_sampling import *
from Util.qm import QueryManager

import pdb

def get_args():
    parser = argparse.ArgumentParser()

    # privacy params
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--dataset', type=str, help='queries', default='adult')
    parser.add_argument('--marginal', type=int, help='queries', default=3)
    parser.add_argument('--workload', type=int, help='queries', default=128)
    parser.add_argument('--workload_seed', type=int, default=0)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--epsilon', type=float, help='Privacy parameter', default=0.1)
    # public dataset params
    parser.add_argument('--dataset_pub', type=str, default=None)
    parser.add_argument('--pub_frac', type=float, default=1.0)
    parser.add_argument('--frac_seed', type=int, default=0)
    # misc params
    parser.add_argument('--permute', action='store_true')
    # bias params
    parser.add_argument('--bias_attr', type=str, default='sex')
    parser.add_argument('--perturb', type=float, default=0)
    # acs params
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--state_pub', type=str, default=None)
    # adult params
    parser.add_argument('--adult_seed', type=int, default=0)

    args = parser.parse_args()
    if args.dataset_pub is None:
        args.dataset_pub = args.dataset

    # validate params
    if args.dataset != 'adult':
        print("This file (specifcally its data loading) is setup for ADULT only.")
        exit()
    if args.dataset != 'adult' and args.pub_frac == 1 and args.frac_seed != 0:
        print("Only need to run frac_seed=0 for pub_frac=1.0")
        exit()
    if args.dataset.startswith('acs_'):
        assert(args.state is not None)
        assert(args.state_pub is not None)

    print(args)
    return args

if __name__ == "__main__":
    args = get_args()

    dataset_name = args.dataset
    if args.dataset.startswith('acs_') and args.state is not None:
        dataset_name += '_{}'.format(args.state)
    results_dir ='results/{}'.format(dataset_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    save_dir_query = 'save/qm/{}/{}_{}_{}/'.format(dataset_name, args.marginal, args.workload, args.workload_seed)
    save_dir_xy = save_dir_query + 'mwem_pub_bias/{}_{}_{}_{}_{}'.format(args.dataset_pub, args.state_pub, args.pub_frac, args.frac_seed, args.perturb)
    for d in [save_dir_query, save_dir_xy]:
        if not os.path.exists(d):
            os.makedirs(d)

    proj = get_proj(args.dataset)
    if args.dataset.endswith('-small'):
        if args.dataset.startswith('acs'):
            args.dataset = args.dataset[:-6]
            args.dataset_pub = args.dataset_pub[:-6]

    filter_private, filter_pub = get_filters(args)

    data, workloads = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj,
                                 filter=filter_private, args=args)
    query_manager = QueryManager(data.domain, workloads)
    N = data.df.shape[0]

    data_all, _ = randomKway(args.dataset, args.workload, args.marginal, seed=args.workload_seed, proj=proj, args=args)
    N_pub = int(N * args.pub_frac)
    data_pub, A_init, orig_attr_distr = get_pub_dataset_biased(data_all, args.pub_frac, args.frac_seed, args.bias_attr, args.perturb)

    print('workload: ', len(workloads))
    print('num queries: ', query_manager.num_queries)
    print('A:', A_init.shape)

    # get answers and initial error
    real_answers = query_manager.get_answer(data)
    query_manager.setup_query_attr(save_dir=save_dir_query)
    query_manager.setup_xy(data_pub, save_dir=save_dir_xy)
    fake_answers = query_manager.get_answer_weights(A_init)
    init_error = np.abs(real_answers - fake_answers).max()

    delta = 1.0 / N ** 2
    rho = cdp_rho(args.epsilon, delta)
    eps0 = (2 * rho) ** 0.5 / (2 * args.T) ** 0.5

    result_cols = {'adult_seed': args.adult_seed,
                   'marginal': args.marginal,
                   'num_workloads': len(workloads),
                   'workload_seed': args.workload_seed,
                   'num_queries': query_manager.num_queries,
                   'dataset_pub': args.dataset_pub,
                   'state_pub': args.state_pub,
                   'pub_frac': args.pub_frac,
                   'priv_size': N,
                   'pub_size': N_pub,
                   'bias_attr': args.bias_attr,
                   'orig_attr_distr': orig_attr_distr[0],
                   'perturb': args.perturb
                   }
    for _ in range(args.num_runs):
        run_id = hash(time.time())
        A_avg, A_last, A_noisy_best = generate(data_pub, real_answers, A_init, query_manager, N, args.T, eps0,
                                               permute=args.permute)

        fake_answers = query_manager.get_answer_weights(A_avg)
        error = np.abs(real_answers - fake_answers).max()
        max_error_avg = error

        fake_answers = query_manager.get_answer_weights(A_last)
        error = np.abs(real_answers - fake_answers).max()
        max_error_last = error

        fake_answers = query_manager.get_answer_weights(A_noisy_best)
        error = np.abs(real_answers - fake_answers).max()
        max_error_noisy_best = error

        results = {'run_id': [run_id],
                   'epsilon': [args.epsilon],
                   'init_error': [init_error],
                   'permute': [args.permute],
                   'T': [args.T],
                   'rho': [rho],
                   'eps0': [eps0],
                   'max_error_avg': max_error_avg,
                   'max_error_last': max_error_last,
                   'max_error_noisy_best': max_error_noisy_best,
                   }
        df_results = pd.DataFrame.from_dict(results)
        i = df_results.shape[1]

        for key, val in result_cols.items():
            df_results[key] = val

        # rearrange columns for better presentation
        cols = list(df_results.columns[i:]) + list(df_results.columns[:i])
        df_results = df_results[cols]

        # only need these cols for ACS experiments
        if args.state_pub is None:
            del df_results['dataset_pub']
            del df_results['state_pub']

        # save results
        results_path = os.path.join(results_dir, 'pmw_pub_biased.csv')
        save_results(df_results, results_path=results_path)
        