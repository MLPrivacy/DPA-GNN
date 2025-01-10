import torch.nn as nn
import torch.nn.functional as F
from layer import MLP


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = MLP(nfeat, nhid)
        self.gc2 = MLP(nhid, nclass)
        self.dropout = dropout


    def forward(self, x, edges):
        x = F.relu(self.gc1(x, edges))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edges)
        return x



# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout):
#         super(GCN, self).__init__()
#         self.fc1 = nn.Linear(nfeat, nhid)
#         self.fc2 = nn.Linear(nhid, nclass)
#         self.dropout = dropout
#
#     def forward(self, x, edges):
#         x = F.relu(self.fc1(x.float()))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.fc2(x.float())
#         return F.log_softmax(x, dim=1)


