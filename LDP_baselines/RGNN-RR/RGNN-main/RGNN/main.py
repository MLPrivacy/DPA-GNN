import sys
import os
import datetime
from typing import Union

import numpy as np
import argparse
import random
import time
from termcolor import cprint
from collections import defaultdict
from copy import deepcopy
import pandas as pd

import torch

from data import load_dataset
from perturb import GRR, GRRFS
from model import SAGE, GCN, GAT
from train import Trainer

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cpu')
def get_arguments():
    parser = argparse.ArgumentParser(description='pyg version of GraphSAGE')

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA')
    parser.add_argument('--num_runs', type=int, default= 10,
                        help='Experiments are run with incremental random seeds')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.25)
    parser.add_argument('--test_ratio', type=float, default=0.25)
    parser.add_argument('--cols_to_group', type=int, default=25,  # cora:25  citeseer:70  Lastfm:80  Facebook:100  amazon :10  reddit:10
                        help='Combine features to reduce sparsity; 0/1=no grouping')

    # GNN
    parser.add_argument('--model', type=str, default='gcn',
                        help='Architecture:sage,gcn,gat')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=16,
                        help='Hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (L2 loss on parameters).')

    # Privacy
    parser.add_argument('--x_eps', type=float, default=0.1,
                        help='Privacy budget for node features')
    parser.add_argument('--m', type=int, default=10,
                        help='No. of features to sample for perturbation;'
                             '0: All Uniform and None: All GRR')
    parser.add_argument('--y_eps', type=float, default=60,
                        help='Privacy budget for node labels')
    parser.add_argument('--e_eps', type=float, default=60,
                        help='Privacy budget for node edges')

    # Reconstruction and LLP
    parser.add_argument('--xhops', type=int, default=8,
                        help='Hops for fx')
    parser.add_argument('--yhops', type=int, default=0,
                        help='Hops for fy')
    parser.add_argument('--num_clusters', type=int, default=16,  # {16,32,64,128,256}
                        help='No. of bags for LLP')
    parser.add_argument('--alpha', type=float, default=1,  # {0.01,0.1,1,10,20}
                        help='LLP regularization hyperparameter')

    parser.add_argument('--gpu_id', type=int, default=0)

    return parser.parse_args()


def pprint(args):
    for k, v in args.__dict__.items():
        print("\t- {}: {}".format(k, v))


def get_model(model_name):
    if str.upper(model_name) == 'SAGE':
        return SAGE
    elif str.upper(model_name) == 'GCN':
        return GCN
    elif str.upper(model_name) == 'GAT':
        return GAT


def run(_args):
    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        gpu_id = int(_args.gpu_id) if _args.gpu_id is not None else 1
        torch.cuda.set_device(gpu_id)
        _args.device = torch.device('cuda:{}'.format(gpu_id))
        print('Using GPU with ID:{}'.format(torch.cuda.current_device()))
    else:
        print("CUDA not available!")

    random.seed(_args.seed)
    np.random.seed(_args.seed)
    torch.manual_seed(_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_args.seed)

    data = load_dataset(_args.dataset, _args.val_ratio, _args.test_ratio, cols_to_group=_args.cols_to_group,
                        e_eps=_args.e_eps)
    _args.num_features = data.num_features
    _args.num_classes = data.num_classes


    # Apply perturbations on features
    Mx = GRRFS(x_eps=_args.x_eps, m=args.m)
    data = Mx(data)

    # Apply perturbations on labels
    My = GRR(_args.y_eps)
    data = My(data)

    gnn_cls = get_model(_args.model)
    gnn = gnn_cls(_args=_args, in_channels=_args.num_features, hidden_channels=_args.hidden,
                  out_channels=_args.num_classes)

    trainer = Trainer(_args, gnn)
    data = data.to(device)
    results = trainer.fit(data)

    fmt_results = results['test_at_best_val']

    with open(os.path.join('./results', _args.out_file), 'a') as f:
        f.write('seed,model,x_eps,y_eps,xhops,m,yhops,clusters,alpha,avg_test_acc,{},{},{},{},{},{},{},{},{},{}\n'.
                format(_args.seed,_args.model, _args.x_eps, _args.y_eps, _args.xhops, _args.m, _args.yhops,
                       _args.num_clusters, _args.alpha, fmt_results))

    return results


def run_multiple(args):
    results = defaultdict(list)
    for i in range(args.num_runs):
        cprint("## TRIAL {} ##".format(i+1), "yellow")
        _args = deepcopy(args)
        _args.seed = _args.seed + i
        # print(_args.seed)
        ret = run(_args)
        _args.seed = args.seed + i
        for rk, rv in ret.items():
            results[rk].append(rv)

    return results


def summarize_results(results):
    cprint("## RESULTS SUMMARY ##", "yellow")

    res_table = {}
    for rk, rv in sorted(results.items()):
        if isinstance(rv, list):
            res_table[rk] = "{} +- {}".format(round(float(np.mean(rv)), 1), round(float(np.std(rv)), 2))
        else:
            res_table[rk] = f'{round(float(rv), 1)}'

    df = pd.Series(res_table).to_frame()
    df.columns = ['avg+-std']
    # print(df)

    # return res_table['test_at_best_val']
    return df


if __name__ == '__main__':
    args = get_arguments()
    pprint(args)

    if args.out_file is None:
        date = '{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        args.out_file = "{}_{}.txt".format(args.dataset, date)

    t0 = time.perf_counter()

    # res = run_multiple(args)
    # summary_test_acc = summarize_results(res)

    print("dataset：", args.dataset)
    with open('output.txt', 'w') as f:
        print("dataset：", args.dataset, file=f)


    for all_eps in [10]:  # 2,4,6,8,10
        for y_eps in [1]:  # 1,1.5,2,2.5
            for eps_rate in [0.05]:  # 0.05,0.2
                x_eps = all_eps * eps_rate
                e_eps = all_eps - x_eps - y_eps
                args.x_eps = x_eps
                args.y_eps = y_eps
                args.e_eps = e_eps


                res = run_multiple(args)
                summary_test_acc = summarize_results(res)

                print("Results for all_eps={}, eps_rate={}, x_eps={}, y_eps={}, e_eps={}".format(all_eps, eps_rate, x_eps, y_eps, e_eps))
                print(summary_test_acc)
                with open('output.txt', 'a') as f:
                    print(args, file=f)
                    print("Results for all_eps={}, eps_rate={}, x_eps={}, y_eps={}, e_eps={}".format(all_eps, eps_rate, x_eps, y_eps, e_eps), file=f)
                    print(summary_test_acc, file=f)
                    print("***************************************************************", file=f)
                    print("***************************************************************", file=f)


    cprint("Time for runs (s): {}\n".format(time.perf_counter() - t0), "green")

