################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        
        gaussian = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
        
        self.W_hx = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_hidden,input_dim))).view(num_hidden,input_dim))
        self.W_hh = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_hidden,num_hidden))))
        self.W_ph = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_classes,num_hidden))))
        
        self.b_h = torch.nn.Parameter(torch.ones([num_hidden,1]))
        self.b_p = torch.nn.Parameter(torch.ones([num_classes,1]))
        
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=0)
        
    def forward(self, x):
        # Implementation here ...
        
        h_prev = torch.zeros([self.num_hidden,self.batch_size])
        for i in range(self.seq_length):
            x_t = x[:,i].view(1,x[:,i].shape[0])
     
            h_t = self.tanh(torch.mm(self.W_hx,x_t) + torch.mm(self.W_hh,h_prev) + self.b_h)
 
            y_t = torch.mm(self.W_ph,h_t) + self.b_p
            
            h_prev = h_t
        
        return y_t
            
        
        
        
            
        
        
        
        
        
        
