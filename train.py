from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy,load_dataset
from model import GCN
import warnings
import random
import statistics

torch.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


parser.add_argument('--train_rate', type=float, default=0.5,
                    help='train_rate.')
parser.add_argument('--val_rate', type=float, default=0.75,
                    help='val_rate.')

# parser.add_argument('--dummy_r', type=float, default=0.1,
#                     help='the r of dummy.')
# parser.add_argument('--c_rate', type=float, default=0.8,
#                     help='the percentage of clipping.')
# parser.add_argument('--epsilon_x', type=float, default=1,
#                     help='epsilon of feature DP Gaussian.')
# parser.add_argument('--epsilon_y', type=float, default=3,
#                     help='epsilon of label DP Gaussian.')
# parser.add_argument('--layerx', type=float, default=4,
#                     help='the layers of feature.')
# parser.add_argument('--layery', type=float, default=4,
#                     help='the layers of label.')

"Parameter setting"
parser.add_argument('--run_times', type=float, default= 10,
                    help='run_times.')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset name (cora, citeseer, lastfm, facebook,amazon,reddit)')

Dummy_r = [0.5]
rate_clipping = [0.8]     # cora 0.8, citeseer 0.8, lastfm 0.7  facebook 0.6, amazon 0.4, reddit 0.2
# eps_x = [2]
# eps_y = [4]
eps_all = [10]  #2,4,6,8,10
eps_rate = [0.05]  #0.05, 0.2
layerx = [10]    # cora [10,8], citeseer [8,10], lastfm [10,6], facebook [6,4], amazon [10,2], reddit [8,2]
layery = [8]



args = parser.parse_args(args=[])
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
# print(args.cuda)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cpu')



random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


with open('output.txt', 'w') as f:
    print("parameter :", args, file=f)


def run(dummy_r, c_rate,epsilon_x,epsilon_y,layerx,layery,dataset):

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("begin_time:", formatted_time)

    sum_1, edges, labels_truth, labels_noise, degree, idx_train, idx_val, idx_test = load_dataset(
        dummy_r=dummy_r,
        eps_x=epsilon_x,
        eps_y=epsilon_y,
        rate=c_rate,
        train_rate=args.train_rate,
        val_rate=args.val_rate,
        layerx=layerx,
        layery=layery,
        dataset_name = dataset)


    # Model and optimizer
    model = GCN(nfeat=sum_1.shape[1],
                nhid=args.hidden,
                nclass=labels_truth.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)


    if args.cuda:
        model.cuda()
        sum_1 = sum_1.cuda()
        edges = edges.cuda()
        labels_noise = labels_noise.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    def train(labels_noise):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output1 = model(sum_1,edges)
        output = F.log_softmax(output1, dim=1)
        labels_noise = torch.argmax(labels_noise, dim=1)
        output = output.to(device)
        loss_train = F.nll_loss(output[idx_train], labels_noise[idx_train].long())
        acc_train = accuracy(output[idx_train], labels_noise[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            model.eval()
            output = model(sum_1,edges)
            output = F.log_softmax(output, dim=1)

        loss_val = F.nll_loss(output[idx_val], labels_noise[idx_val].long())
        acc_val = accuracy(output[idx_val], labels_noise[idx_val])

        # print('Epoch: {:04d}'.format(epoch+1),
        #       "Train set results:",
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #       'acc_train: {:.4f}'.format(acc_train.item()),
        #       "Validation set results:",
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'acc_val: {:.4f}'.format(acc_val.item()))

        return loss_val,output1


    def end_result():
        model.eval()
        output = model(sum_1,edges)
        output = F.log_softmax(output, dim=1)
        loss_test = F.nll_loss(output[idx_test], labels_truth[idx_test].long())
        acc_test = accuracy(output[idx_test], labels_truth[idx_test])

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return acc_test, output[idx_test], labels_truth[idx_test]


    t_total = time.time()
    eval_T = 5  # evaluate period
    P = 9  # patience
    i = 0  # record the frequency of bad performance of validation
    temp_val_loss = 99999  # initialize val loss

    for epoch in range(args.epochs):
        result, output1= train(labels_noise)
        # early stopping
        if (epoch % eval_T) == 0:
            if temp_val_loss > result:
                temp_val_loss = result
                # torch.save(model.state_dict(), "GCN_NET3.pth")  # save the current best
                i = 0  # reset i
            else:
                i = i + 1
        if i > P:
            print("Early Stopping! Epoch1 : ", epoch )
            break

    test_acc, result, test_label = end_result()

    current_time = datetime.now()
    end_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time:", end_time)


    return test_acc



for dummy_r in Dummy_r:
    for c_rate in rate_clipping:
        # for epsilon_x in eps_x:
        #     for epsilon_y in eps_y:
        for epsilon_all in eps_all:
            for epsilon_rate in eps_rate:
                for layer_x in layerx:
                    for layer_y in layery:
                        epsilon_x = epsilon_all * epsilon_rate
                        epsilon_y = epsilon_all - epsilon_x - 1.25
                        print(f"--------------------------------------Processing with dummy_r: {dummy_r}--------------------------------------")
                        print(f"--------------------------------------Processing with c_rate: {c_rate}--------------------------------------")
                        print(f"--------------------------------------Processing with epsilon_x: {epsilon_x}--------------------------------------")
                        print(f"--------------------------------------Processing with epsilon_y: {epsilon_y}--------------------------------------")
                        print(f"--------------------------------------Processing with features layer: {layer_x} --------------------------------------")
                        print(f"--------------------------------------Processing with label layer: {layer_y}--------------------------------------")

                        with open('output.txt', 'a') as f:
                            print(f"--------------------------------------Processing with dummy_r: {dummy_r}--------------------------------------",file=f)
                            print(f"--------------------------------------Processing with c_rate: {c_rate}--------------------------------------",file=f)
                            print(f"--------------------------------------Processing with epsilon_x: {epsilon_x}--------------------------------------",file=f)
                            print(f"--------------------------------------Processing with epsilon_y: {epsilon_y}--------------------------------------",file=f)
                            print(f"--------------------------------------Processing with features layer: {layer_x}--------------------------------------",file=f)
                            print(f"--------------------------------------Processing with label layer: {layer_y}--------------------------------------",file=f)

                        run_times = args.run_times
                        results_all = [(print(f"-----------------------Run {_ + 1}/{args.run_times}---------------------------"),
                                        run(dummy_r, c_rate, epsilon_x, epsilon_y, layer_x, layer_y,
                                            args.dataset))[1]
                                       for _ in range(args.run_times)]
                        regular_list = [tensor.cpu().item() for tensor in results_all]
                        print("Function return value list:", regular_list)
                        average = statistics.mean(regular_list)
                        std_dev = statistics.stdev(regular_list)
                        print("Average + standard deviation:", round(average * 100, 1), '+-', round(std_dev * 100, 2))


                        with open('output.txt', 'a') as f:
                            print("Function return value list:", regular_list, file=f)
                            print("Average + standard deviation:", round(average * 100, 1), '+-',
                                  round(std_dev * 100, 2), file=f)




