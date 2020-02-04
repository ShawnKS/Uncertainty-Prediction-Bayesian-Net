import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, GAT_LSTM
from torch.nn import init

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        print('lstm01')
        self.lstm01 = GAT_LSTM(input_size=12, hidden_size=32, nodes_number=228)
        # for weight in self.lstm01.parameters():
        #     init.constant_(weight, 0.5)
        print('lstm02')
        self.lstm02 = GAT_LSTM(input_size=32, hidden_size=128, nodes_number=228)
        # for weight in self.lstm02.parameters():
        #     init.constant_(weight, 0.5)
        self.reg = nn.Linear(128, 9)
        self.batch_norm = nn.BatchNorm1d(12, affine=True)

    # def reset_weigths(model):
    #     """reset weights
    #     """
    #     for weight in model.parameters():
    #         init.constant_(weight, 0.5)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj), inplace=True)
        x = x.float()
        T, N, Fea = x.shape  # T=Length_of_sequence, N=No.of_nodes, F=No.of_features

        #batchnorm
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)

        x = x.permute(1, 0, 2)
        #lstm01
        # h0 = torch.ones(1, 64, 228)
        # c0 = torch.ones(1, 64, 228)
        x, _ = self.lstm01(x, None)

        x = x.permute(1, 0, 2)
        #lstm02
        # h0 = torch.ones(1, 128, 228)
        # c0 = torch.ones(1, 128, 228)
        x, _ = self.lstm02(x, None)
        x = x.reshape(-1,1,128)
        x = self.reg(x)
        x = x.view(T, N, -1)  # (7580, 228, 3)
        return x



# Original GAT module
# class GAT(nn.Module):
#     def __init__(self, nfeat, nhid, n_out, dropout, alpha, nheads):
#         """Dense version of GAT."""
#         super(GAT, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         self.out_att = GraphAttentionLayer(nhid * nheads, n_out, dropout=dropout, alpha=alpha, concat=False)
#         self.out_lstm = LstmLayer(12, 24, output_size=9, num_layers=1)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj), inplace=True)
#         x = self.out_lstm(x)
#         return x



# class SpGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
#         """Sparse version of GAT."""
#         super(SpGAT, self).__init__()
#         self.dropout = dropout
#
#         self.attentions = [SpGraphAttentionLayer(nfeat,
#                                                  nhid,
#                                                  dropout=dropout,
#                                                  alpha=alpha,
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)
#
#         self.out_att = SpGraphAttentionLayer(nhid * nheads,
#                                              nclass,
#                                              dropout=dropout,
#                                              alpha=alpha,
#                                              concat=False)
#
#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return F.log_softmax(x, dim=1)

