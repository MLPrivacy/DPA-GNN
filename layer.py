import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cpu')


class MLP(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)



    def forward(self, input_fea, edges):
        sum_1 = input_fea.to(self.weight.dtype)
        sum_1 = sum_1.to(device)
        self.weight = self.weight.to(device)
        output = torch.mm(sum_1, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

