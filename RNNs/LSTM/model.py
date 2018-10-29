# MIT License
#
# Copyright (c) 2017 Tom Runia
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

import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_dim = lstm_num_hidden
        self.onehot_input = torch.Tensor(self.batch_size,vocabulary_size)
        self.onehot_input_eval = torch.Tensor(1,vocabulary_size)
        self.vocabulary_size = vocabulary_size
        
        self.lstm_2_layers = nn.LSTM(vocabulary_size, lstm_num_hidden, num_layers=2)
        
        self.fc = nn.Linear(lstm_num_hidden, vocabulary_size)

        self.softmax = nn.Softmax(dim=2)
        

    def forward(self, x,cuda,temp):
        
        dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        # Implementation here...
        init_state = self.init_hidden_cell(self.batch_size,dtype)
        
        model_input=torch.zeros(self.seq_length,self.batch_size,self.vocabulary_size)
        
        for t in range(self.seq_length):
            self.onehot_input.zero_()
            input_ = x[t].view(x[t].shape[0],1)
            input_= self.onehot_input.scatter_(1, input_ , 1).view(1,self.batch_size,self.onehot_input.shape[1])
            model_input[t,:,:] = input_
    

        out , (h_n,c_n) = self.lstm_2_layers(model_input.type(dtype),s)
        
        y = torch.div(self.fc(out),temp)
        
        return y
    
    def init_hidden_cell(self,batch_size,dtype):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, batch_size, self.hidden_dim).type(dtype),
                torch.zeros(2, batch_size, self.hidden_dim).type(dtype))
    
    
    def random_sampling(self,text_size,cuda,temp):

        dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        (h_n,c_n) = self.init_hidden_cell(1,dtype)
        
        input_ = torch.Tensor([np.random.randint(self.vocabulary_size)])
        
        text = torch.zeros(text_size)
        
        y_=0
        for i in range(text_size):
            self.onehot_input_eval.zero_()
            
            if i != 0:
                input_ = torch.argmax(y_[1,0,:]).view(1,1)
            else :
                input_ = input_.view(1,1)
#            
            text[i] = input_
            
            input_= self.onehot_input_eval.scatter_(1, input_.type(torch.LongTensor) , 1).view(1,1,self.onehot_input.shape[1])
            
            out, (h_n,c_n) = self.lstm_2_layers(input_.type(dtype),(h_n,c_n))
            
            y_ = self.softmax(torch.div(self.fc(h_n),temp))
        
        return text
    
    def create_text(self,x,text_size,cuda,temp):
        
        dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        
        init_state = self.init_hidden_cell(1,dtype)
        
        model_input=torch.zeros(self.seq_length,x.shape[1],self.vocabulary_size)
        
        text = torch.zeros(text_size+self.seq_length)

        text[0:self.seq_length] = x[:,0]
        
        for t in range(self.seq_length):
            self.onehot_input_eval.zero_()
            
            input_ = x[t].view(x[t].shape[0],1)
            
            input_= self.onehot_input_eval.scatter_(1, input_ , 1).view(1,1,self.onehot_input.shape[1])

            model_input[t,:,:] = input_
        
        out , (h_n,c_n) = self.lstm_2_layers(model_input.type(dtype),init_state)
        
        y_ = self.softmax(torch.div(self.fc(h_n),temp))

        
        for i in range(text_size):
            self.onehot_input_eval.zero_()

            input_ = torch.argmax(y_[1,0,:]).view(1,1)
            
            text[i+self.seq_length] = input_
            
            input_= self.onehot_input_eval.scatter_(1, input_.type(torch.LongTensor) , 1).view(1,1,self.onehot_input.shape[1])
            
            out, (h_n,c_n) = self.lstm_2_layers(input_.type(dtype),(h_n,c_n))
            
            y_ = self.softmax(torch.div(self.fc(h_n),temp))

        return text
        
           
        
        
            
        
