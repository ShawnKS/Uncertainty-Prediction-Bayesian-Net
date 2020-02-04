# GAT-TSP

import numpy as np
import scipy.sparse as sp
import torch
import csv
from math_utils import *
from scipy.sparse.linalg import eigs
import random
import networkx as nx
import matplotlib.pyplot as plt


def seq_gen(data_seq, n_frame):
    '''
    Generate data in the form of standard sequence unit.

    Parameters
    ----------
    data_seq: np.ndarray, time-series, shape is (length, num_of_vertices)


    n_frame: int, n_his + n_pred

    Returns
    ----------
    np.ndarray, shape is (length - n_frame + 1, n_frame, num_of_vertices, 1)

    '''

    data = np.zeros(shape=(data_seq.shape[0] - n_frame + 1,
                           n_frame, data_seq.shape[1], 1))
    for i in range(data_seq.shape[0] - n_frame + 1):
        data[i, :, :, 0] = data_seq[i: i + n_frame, :]
    return data


def data_gen(file_path, n_frame=576):
    '''
    Source file load and dataset generation.

    Parameters
    ----------
    file_path: str, path of time series data

    n_frame: int, n_his + n_pred

    Returns
    ----------
    Dataset, dataset that contains training, validation and test with stats.

    '''

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data_seq = np.array([list(map(float, i)) for i in reader if i])  # (12672, 228)
        features = data_seq  # (228, 12672)


    # generate noisy & missing test set
    Noisy = False
    Missing = True
    if Noisy:
        features = gen_noisy_data(features, 0.005)
    if Missing:
        features = gen_missing_data(features, 0.005)


    #convert into tensors
    # features = torch.from_numpy(features)

    num_of_samples = features.shape[0]
    splitting_line1 = int(num_of_samples * 0.6)
    splitting_line2 = int(num_of_samples * 0.8)

    #split the dataset
    data_seq = features
    seq_train = seq_gen(data_seq[: splitting_line1], n_frame)
    # print('seq_train shape', seq_train.shape)
    seq_val = seq_gen(data_seq[splitting_line1: splitting_line1+576], n_frame)
    seq_test = seq_gen(data_seq[splitting_line2:], n_frame)

    #Z-score normalization
    mean = np.mean(seq_train)
    std = np.std(seq_train)
    x_train = z_score(seq_train, mean, std)  # (7580, 24, 228, 1)
    x_val = z_score(seq_val, mean, std)  # (2511, 24, 228, 1)
    x_test = z_score(seq_test, mean, std)  # (2512, 24, 228, 1)

    #generate dataset
    train = x_train.transpose((0, 3, 1, 2))
    val = x_val.transpose((0, 3, 1, 2))
    test = x_test.transpose((0, 3, 1, 2))
    #x (7580, 1, 12, 228)     y (7580, 1, 3, 228)
    n_his = 12
    n_pre = 9  # n_pre = 3, 6, 9 for 15min, 30min, 45min
    train_x, train_y = train[:, :, : n_his, :], train[:, :, n_his:n_his+n_pre, :]
    val_x, val_y = val[:, :, : n_his, :], val[:, :, n_his:n_his+n_pre, :]
    test_x, test_y = test[:, :, : n_his, :], test[:, :, n_his:n_his+n_pre, :]  #

    #visualize forecasting results
    #the second day
    # visual_x, visual_y = train[276:564, :, : n_his, :], train[276:564, :, n_his:n_his+n_pre, :]
    # insert_x = np.array()
    # for i in range(24):
    #     visual_x = visual_x

    #transpose, squeeze
    train_x = train_x.transpose((0, 3, 2, 1)).squeeze(3)
    val_x = val_x.transpose((0, 3, 2, 1)).squeeze(3)
    print('val_x', val_x.shape)
    test_x = test_x.transpose((0, 3, 2, 1)).squeeze(3)
    train_y = train_y.transpose((0, 3, 2, 1)).squeeze(3)
    val_y = val_y.transpose((0, 3, 2, 1)).squeeze(3)
    test_y = test_y.transpose((0, 3, 2, 1)).squeeze(3)


    x_dataset = {'train': train_x, 'val': val_x, 'test': test_x}
    y_dataset = {'train': train_y, 'val': val_y, 'test': test_y}


    print(train_x.shape, train_y.shape, val_x.shape,
          val_y.shape, test_x.shape, test_y.shape)



    return x_dataset, y_dataset, mean, std, train


def gen_noisy_data(data, ratio=0.05):
    # generate the Gaussian noise matrix
    means = np.mean(data)
    gaussian_noise = np.random.normal(0.0, float(means*ratio), size=(data.shape[0], data.shape[1]))
    print(gaussian_noise)
    noisy_data = data + gaussian_noise
    print(data == noisy_data)
    return noisy_data


def gen_missing_data(data, ratio=0.05):
    # generate the Gaussian noise matrix
    initial_0 = data.shape[0]
    initial_1 = data.shape[1]
    data_number = initial_0 * initial_1
    data = data.reshape(-1, 1).squeeze(1)
    converted_list = []
    for i in range(int(data_number*ratio)):
        rand = random.randint(1, data_number)
        while(rand in converted_list):
            rand = random.randint(1, data_number)
        converted_list.append(rand)
        data[rand] = 0
    data = data.reshape(initial_0, initial_1)
    print(len(data[data == 0]))
    missing_data = data
    return missing_data


def adj_gen(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        adj = np.array([list(map(float, i)) for i in reader if i])
    #check whether adj is a 0/1 matrix.
    # if set(np.unique(adj)) == {0, 1}:
    #     print('The input graph is a 0/1 matrix; set "scaling" to False.')
    #     scaling = False
    #
    # if scaling:
    #     adj = adj / 10
    #     mask = np.ones_like(adj) - np.identity(adj.shape[0])
    #     # refer to Eq.10
    #     exp = np.exp(- adj ** 2 / sigma2)
    #     return exp * (exp >= epsilon) * mask

    adj = normalize_adj(adj+np.eye(adj.shape[0]))
    return adj


def k_nearest_adj_gen(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        adj = np.array([list(map(float, i)) for i in reader if i])

    adj_zeros = np.zeros_like(adj)
    #sort the top5 nearest
    for i,j in enumerate(adj):
        j = np.argsort(j, axis=-1, kind='heapsort')
        top_5 = j[:100]
        for k in top_5:
            adj_zeros[i][k] = 1
    adj = adj_zeros
    adj = adj + adj.T*(adj.T > adj) - adj*(adj.T > adj)
    adj = normalize_adj(adj + np.eye(adj.shape[0]))

    return adj


def scaled_laplacian(W):
    '''
    Normalized graph Laplacian

    Parameters
    ----------
    W: np.ndarray, adjacency matrix,
       shape is (num_of_vertices, num_of_vertices)

    Returns
    ----------
    np.ndarray, shape is (num_of_vertices, num_of_vertices)

    '''

    num_of_vertices = W.shape[0]
    d = np.sum(W, axis=1)
    L = np.diag(d) - W
    for i in range(num_of_vertices):
        for j in range(num_of_vertices):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return 2 * L / lambda_max - np.identity(num_of_vertices)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)




# #generate the graph
G = nx.Graph()

print(len(adj))

for i in range(len(adj)):
    for j in range(len(adj)):
        if not adj[i][j] == 0:
            G.add_edge(i, j)

print(G.size())

