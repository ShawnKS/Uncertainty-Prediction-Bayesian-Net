# GAT-TSP
# 显存不够，要用mini-batch了 最大batch_size 为1000

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn import Parameter
import math
from torch import typename

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  # 权值矩阵 #(288, 12)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # Xavier 初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))  # a是一个单层的前馈网络  #(24, 1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # Xavier 初始化

        self.leakyrelu = nn.LeakyReLU(self.alpha, inplace=True)
        self.batch_norm = nn.BatchNorm1d(228, affine=True)

        #注意力系数


    def forward(self, input, adj):  # input (7580, 228, 288)
        # print('self.W', typename(self.W))
        # print('self.W', self.W.shape)
        input = input.float()
        h = torch.matmul(input, self.W)  # (7580, 228, 12)
        N = h.size()[1]  # N 为节点个数 h= {h1,h2,.....hn} 228
        # tmp_a = h.repeat(1, 1, N).view(h.shape[0], N * N, -1).half()
        # tmp_b = h.repeat(1, N, 1).half()
        # a_input = torch.cat([tmp_a, tmp_b], dim=2)
        # del tmp_a, tmp_b
        # a_input = a_input.view(h.shape[0], N, -1, 2 * self.out_features).float()
        a_input = torch.cat([h.repeat(1, 1, N).view(h.shape[0], N * N, -1), h.repeat(1, N, 1)], dim=2).view(h.shape[0], N, -1, 2 * self.out_features).float()  # (7580, 228, 228, 24)
        # e = self.leakyrelu(torch.matmul(a_input, self.a), inplace=True)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # attention coefficients 注意力系数  (7580, 228, 228)


        zero_vec = -9e15*torch.ones_like(e)  # ？？？

        attention = torch.where(adj > 0, e, zero_vec)  # (7580, 228, 228)
        # print(attention[4]==attention[4].transpose(1,0))

        attention = F.dropout(attention, self.dropout, training=self.training)  # 这边采用了dropout机制提升泛化效果

        #Softmax
        attention = F.softmax(attention, dim=2)  # αij

        # 储存attention adjacency map
        # np.save('attention coefficients/e.npy', attention.detach().cpu().numpy())


        h_prime = torch.matmul(attention, h)  # 预测每个节点的output feature h_prime (7580, 228, 12 )
        # h_prime = h_prime.cuda(1)
        if self.concat:
            return self.batch_norm(F.elu(h_prime))
        else:
            return self.batch_norm(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT_LSTM(nn.Module):
    """A rewritten LSTM like nn.LSTM for spatiotemporal input"""

    def __init__(self, input_size: int, hidden_size: int, nodes_number: int):
        super(GAT_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nodes_number = nodes_number
        # self.batch_norm = nn.BatchNorm1d(228)

        # input gate
        self.w_ii = Parameter(Tensor(hidden_size, input_size))
        nn.init.xavier_uniform_(self.w_ii.data, gain=1.414)
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.w_hi.data, gain=1.414)
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.b_ii.data, gain=1.414)
        self.b_hi = Parameter(Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.b_hi.data, gain=1.414)

        # forget gate
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        nn.init.xavier_uniform_(self.w_if.data, gain=1.414)
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.w_hf.data, gain=1.414)
        self.b_if = Parameter(Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.b_if.data, gain=1.414)
        self.b_hf = Parameter(Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.b_hf.data, gain=1.414)

        # output gate
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        nn.init.xavier_uniform_(self.w_io.data, gain=1.414)
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.w_ho.data, gain=1.414)
        self.b_io = Parameter(Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.b_io.data, gain=1.414)
        self.b_ho = Parameter(Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.b_ho.data, gain=1.414)

        # cell
        self.w_ig = Parameter(Tensor(hidden_size, input_size))
        nn.init.xavier_uniform_(self.w_ig.data, gain=1.414)
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.w_hg.data, gain=1.414)
        self.b_ig = Parameter(Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.b_ig.data, gain=1.414)
        self.b_hg = Parameter(Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.b_hg.data, gain=1.414)


        # self.reset_weigths()

    # def reset_weigths(self):
    #     """reset weights
    #     """
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in self.parameters():
    #         init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs: Tensor, state):
        """Forward
        Args:
            inputs: [1, 1, input_size]
            state: ([1, 1, hidden_size], [1, 1, hidden_size])
        """


        #         seq_size, batch_size, _ = inputs.size()

        if state is None:
            h_t = torch.zeros(1, self.hidden_size, self.nodes_number).permute(2, 1, 0).cuda(1)
            c_t = torch.zeros(1, self.hidden_size, self.nodes_number).permute(2, 1, 0).cuda(1)
        else:
            (h, c) = state
            # print('h', h.shape)
            h_t = h.permute(2, 1, 0)
            c_t = c.permute(2, 1, 0)

        if typename(self.w_ii) == 'torch.FloatTensor':
            h_t = h_t.cpu()
            c_t = c_t.cpu()


        hidden_seq = []

        seq_size = inputs.shape[1]
        for t in range(seq_size):
            x = inputs[:, t, :].unsqueeze(2)

            # input gate  '@' means matrix production
            i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t +
                              self.b_hi)
            # forget gate
            f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t +
                              self.b_hf)
            # cell
            g = torch.tanh(self.w_ig @ x + self.b_ig + self.w_hg @ h_t
                           + self.b_hg)
            # output gate
            o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t +
                              self.b_ho)

            c_next = f * c_t + i * g
            h_next = o * torch.tanh(c_next)
            c_next_t = c_next.permute(2, 1, 0).unsqueeze(0)
            h_next_t = h_next.permute(2, 1, 0).unsqueeze(0)
            hidden_seq.append(h_next_t)

        hidden_seq = torch.cat(hidden_seq, dim=0).permute(0, 3, 2, 1).squeeze(3).float()
        # print('hidden_seq', hidden_seq)
        return hidden_seq, (h_next_t, c_next_t)


#Original LSTM approach
# class LstmLayer(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size=9, num_layers=1):
#         super(LstmLayer, self).__init__()
#
#         self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
#         self.reg = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = x.float()
#         x, _ = self.rnn(x)  #double
#         s, b, h = x.shape  # (seq, batch, hidden)  (7580, 228, 12)
#         x = x.view(s*b, h)  #
#         x = self.reg(x)
#         x = x.view(s, b, -1)  # (7580, 228, 3)
#         return x




# class SpecialSpmmFunction(torch.autograd.Function):
#     """Special function for only sparse region backpropataion layer."""
#     @staticmethod
#     def forward(ctx, indices, values, shape, b):
#         assert indices.requires_grad == False
#         a = torch.sparse_coo_tensor(indices, values, shape)
#         ctx.save_for_backward(a, b)
#         ctx.N = shape[0]
#         return torch.matmul(a, b)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         a, b = ctx.saved_tensors
#         grad_values = grad_b = None
#         if ctx.needs_input_grad[1]:
#             grad_a_dense = grad_output.matmul(b.t())
#             edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
#             grad_values = grad_a_dense.view(-1)[edge_idx]
#         if ctx.needs_input_grad[3]:
#             grad_b = a.t().matmul(grad_output)
#         return None, grad_values, None, grad_b
#
#
# class SpecialSpmm(nn.Module):
#     def forward(self, indices, values, shape, b):
#         return SpecialSpmmFunction.apply(indices, values, shape, b)
#
#
# class SpGraphAttentionLayer(nn.Module):
#     """
#     Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features, out_features, dropout, alpha, concat=True):
#         super(SpGraphAttentionLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.alpha = alpha
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
#         nn.init.xavier_normal_(self.W.data, gain=1.414)
#
#         self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
#         nn.init.xavier_normal_(self.a.data, gain=1.414)
#
#         self.dropout = nn.Dropout(dropout)
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.special_spmm = SpecialSpmm()
#
#     def forward(self, input, adj):
#         N = input.size()[0]
#         edge = adj.nonzero().t()
#
#         h = torch.mm(input, self.W)
#         # h: N x out
#         assert not torch.isnan(h).any()
#
#         # Self-attention on the nodes - Shared attention mechanism
#         edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
#         # edge: 2*D x E
#
#         edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
#         assert not torch.isnan(edge_e).any()
#         # edge_e: E
#
#         e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1)).cuda())
#         # e_rowsum: N x 1
#
#         edge_e = self.dropout(edge_e)
#         # edge_e: E
#
#         h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
#         assert not torch.isnan(h_prime).any()
#         # h_prime: N x out
#
#         h_prime = h_prime.div(e_rowsum)
#         # h_prime: N x out
#         assert not torch.isnan(h_prime).any()
#
#         if self.concat:
#             # if this layer is not last layer,
#             return F.elu(h_prime)
#         else:
#             # if this layer is last layer,
#             return h_prime
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
