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
import numpy as np
################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        
        gaussian = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([0.1]))
        
        # input modulation gate
        self.W_gx = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_hidden,input_dim))).view(num_hidden,input_dim))
        self.W_gh = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_hidden,num_hidden))))
        
        # input gate
        self.W_ix = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_hidden,input_dim))).view(num_hidden,input_dim))
        self.W_ih = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_hidden,num_hidden))))
        
        # forget gate
        self.W_fx = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_hidden,input_dim))).view(num_hidden,input_dim))
        self.W_fh = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_hidden,num_hidden))))
        
        # output gate
        self.W_ox = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_hidden,input_dim))).view(num_hidden,input_dim))
        self.W_oh = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_hidden,num_hidden))))        
        
        self.W_ph = torch.nn.Parameter(torch.squeeze(gaussian.sample((num_classes,num_hidden))))
        
        # initialize biases
        self.b_g = torch.nn.Parameter(torch.ones([num_hidden,1]))
        self.b_i = torch.nn.Parameter(torch.ones([num_hidden,1]))
        self.b_f = torch.nn.Parameter(torch.ones([num_hidden,1]))
        self.b_o = torch.nn.Parameter(torch.ones([num_hidden,1]))
        self.b_p = torch.nn.Parameter(torch.ones([num_classes,1]))
        
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=0)
        

    def forward(self, x):
        # Implementation here ...
        h_prev = torch.zeros([self.num_hidden,self.batch_size])
        c_prev = torch.zeros([self.num_hidden,self.batch_size])
        for i in range(self.seq_length):
            x_t = x[:,i].view(1,x[:,i].shape[0])
         
            g_t = self.tanh(torch.mm(self.W_gx,x_t) + torch.mm(self.W_gh,h_prev) + self.b_g)
            
            i_t = self.sigmoid(torch.mm(self.W_ix,x_t) + torch.mm(self.W_ih,h_prev) + self.b_i)
            
            f_t = self.sigmoid(torch.mm(self.W_fx,x_t) + torch.mm(self.W_fh,h_prev) + self.b_f)
            
            o_t = self.sigmoid(torch.mm(self.W_ox,x_t) + torch.mm(self.W_oh,h_prev) + self.b_o)
            
            c_t = g_t * i_t + c_prev * f_t
#
            h_t = self.tanh(c_t) * o_t 
            
            y_t = torch.mm(self.W_ph,h_t) + self.b_p
            
#            y_t = self.softmax(p_t)
            
            h_prev = h_t
            
            c_prev = c_t
        
        return y_t