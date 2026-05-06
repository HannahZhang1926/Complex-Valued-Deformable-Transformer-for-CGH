import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import einops

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        self.bias = bias
        if bias:
            self.bias_r = Parameter(torch.Tensor(out_features))
            self.bias_i = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_i', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_r.weight)
        nn.init.xavier_uniform_(self.fc_i.weight)
        if self.bias:
            nn.init.constant_(self.bias_r, 0.)
            nn.init.constant_(self.bias_i, 0.)

    def forward(self, input):
        input_r = input.real
        input_i = input.imag
        if self.bias:
            return torch.complex(self.fc_r(input_r)-self.fc_i(input_i)+self.bias_r, 
                                 self.fc_r(input_i)+self.fc_i(input_r)+self.bias_i)
        else:
            return torch.complex(self.fc_r(input_r)-self.fc_i(input_i), 
                                 self.fc_r(input_i)+self.fc_i(input_r))

        
