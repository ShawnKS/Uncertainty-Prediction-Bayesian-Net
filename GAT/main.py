# GAT-TSP
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import data_gen, adj_gen, k_nearest_adj_gen
from math_utils import *
from math_graph import *
from models import GAT
from torch import typename
from calculate_memory import modelsize

import matplotlib.pyplot as plt


# Training settings
# torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate.')  # default 0.0001
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=12, help='Number of hidden units.')
# parser.add_argument('--out', type=int, default=32, help='Number of output of GAT.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--batch_size', type=int, default=50, help='Number of batch size.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
x_file_path = 'dataset/PeMSD7_V_228.csv'
adj_file_path = 'dataset/PeMSD7_W_228.csv'
x_dataset, y_dataset, mean, std, train_data = data_gen(x_file_path)
adj = adj_gen(adj_file_path)
print(len(adj[adj == 0]))
# Convert into tensor
adj = torch.from_numpy(adj)

train_x = torch.from_numpy(x_dataset['train'])
train_y = torch.from_numpy(y_dataset['train'])
val_x = torch.from_numpy(x_dataset['val'])
val_y = torch.from_numpy(y_dataset['val'])
test_x = torch.from_numpy(x_dataset['test'])
print('test_x', test_x.shape)
test_y = torch.from_numpy(y_dataset['test'])
print('test_y', test_y.shape)

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=train_x.shape[2],
                nhid=args.hidden, 
                n_out=args.out,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=train_x.shape[2],
                nhid=args.hidden, 
                # n_out=args.out,
                dropout=args.dropout,
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)



if args.cuda:
    adj = adj.float().cuda(1)

    train_x = train_x.float().cuda(1)
    val_x = val_x.float().cuda(1)
    test_x = test_x.float().cuda(1)
    train_y = train_y.cuda(1)
    val_y = val_y.float().cuda(1)
    test_y = test_y.float().cuda(1)



def train(epoch):
    t = time.time()
    permutation = torch.randperm(train_x.shape[0])
    epoch_training_losses = []
    global output
    global optimizer

    for i in range(0, train_x.shape[0], args.batch_size):

        model.train()
        optimizer.zero_grad()
        indices = permutation[i:i + args.batch_size]
        batch_x, batch_y = train_x[indices], train_y[indices]
        batch_x = batch_x.cuda(1)
        batch_y = batch_y.cuda(1)
        # model size
        # modelsize(model, batch_x, adj, type_size=4)
        output = model(batch_x, adj)  # adj:Long batch_x:double
        loss_criterion = nn.MSELoss()  # 损失函数NLLLoss() 的 输入 是一个对数概率向量和一个目标标签, 适合最后一层是softmax
        batch_y = batch_y.float()
        # print('output shape', output.shape)
        # print('batch_y sahape', batch_y.shape)
        loss_train = loss_criterion(output, batch_y)
        loss_train = loss_train.float()
        loss_train.backward()
        optimizer.step()
        epoch_training_losses.append(loss_train.detach().cpu().numpy())
        # if not args.fastmode:
        #     # Evaluate validation set performance separately,
        #     # deactivates dropout during validation run.
        #
        #     model.eval()  #
        #     # model.dropout = False
        #     output = model(val_x, adj)
        #     # model.dropout = True

    t_training_end = time.time()
    t = t_training_end - t_training_end

    return sum(epoch_training_losses)/len(epoch_training_losses), t


def compute_test():
    with torch.no_grad():
        model.eval()
        # print('test_x', typename(test_x))
        # print('adj', typename(adj))
        output = model(test_x, adj)
        # loss_test = loss_criterion(output, test_y)
        output_unnormalized = z_inverse(output.detach().cpu().numpy(), mean, std)
        y_test_unnormalized = z_inverse(test_y.detach().cpu().numpy(), mean, std)
        # loss metrics
        mae = MAE(y_test_unnormalized, output_unnormalized)
        rmse = RMSE(y_test_unnormalized, output_unnormalized)
        mape = masked_mape_np(y_test_unnormalized, output_unnormalized, null_val=0)

        print('output_unnormalized', output_unnormalized)
        print('y_test_unnormal', y_test_unnormalized)
        print('MAE', mae)
        print('RMSE', rmse)
        print('MAPE', mape)

        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()))

# Train model
t_total = time.time()
loss_train_list = []
loss_val_list = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
loss_criterion = nn.MSELoss()
print('start training')

for epoch in range(args.epochs):
    model = model.cuda(1)
    adj = adj.cuda(1)
    t_start = time.time()
    loss_train, t = train(epoch)
    loss_train_list.append(loss_train)
    t_end = time.time()
    t_consume = t_end - t_start
    print('每epoch消耗时间为', t_consume)
    #Run validation  可能因为val数据量太大 但是如果小的话
    with torch.no_grad():
        model.eval()
        # model.dropout = False
        print('val_x', val_x.shape)
        output_val = model(val_x, adj)
        loss_val = loss_criterion(output_val, val_y)
        loss_val_list.append(loss_val.detach().cpu().numpy())
        # model.dropout = True


    print('this epoch last for {}'.format(t))
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()))

    #Plot
    # plt.plot(loss_train_list, label="training loss", color='r')
    # plt.plot(loss_val_list, label="validation loss")
    # plt.legend()
    #
    # plt.show()

    # torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    # if loss_values[-1] < best:  # 寻找最好的一次世代结果
    #     best = loss_values[-1]
    #     best_epoch = epoch
    #     bad_counter = 0
    # else:
    #     bad_counter += 1
    #
    # if bad_counter == args.patience:
    #     break
    #
    # files = glob.glob('*.pkl')
    # for file in files:
    #     epoch_nb = int(file.split('.')[0])
    #     if epoch_nb < best_epoch:
    #         os.remove(file)

# files = glob.glob('*.pkl')
# for file in files:
#     epoch_nb = int(file.split('.')[0])
#     if epoch_nb > best_epoch:
#         os.remove(file)


# Store loss numpy
loss_train_numpy = np.array(loss_train_list)
loss_val_numpy = np.array(loss_val_list)
np.save('nl_m0005_45min_loss_train_numpy.npy', loss_train_numpy)
np.save('nl_m0005_45min_loss_val_numpy.npy', loss_val_numpy)


# Store the model
state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':args.epochs}
torch.save(state, 'nl-m0005-45min-500eps.pkl')
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))


# Restore best model
# print('Loading {}th epoch'.format(best_epoch))
# model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))


#Restore the dedicated model
# checkpoint = torch.load('nl-l832-45min-500eps.pkl')
# model.load_state_dict(checkpoint['net'])


# Testing
del train_x, train_y, val_x, val_y, output
test_x = test_x.cpu()
model = model.cpu()
adj = adj.cpu()
compute_test()
