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

#import os
import time
from datetime import datetime
import argparse

#import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

#from part3.dataset import TextDataset
#from part3.model import TextGenerationModel

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def train(config):

    # Initialize the device which to run the model on
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("cuda")
        device = torch.device('cuda:0')
    else:
        print("no cuda")
        device = torch.device('cpu')
    
  
    
    # Text generation options
    generate_text=True
    generated_text_size=1500
    fixed_output_samples=False
    fixed_random_samples=True
    
        
#    device = torch.device(device)
    dtype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file,config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

     # Initialize the model that we are going to use
    if config.load_model == "none" :
        model = TextGenerationModel(config.batch_size, config.seq_length,dataset.vocab_size).to(device)  # fixme
        print(model)
    else :
        print("load model")
        model = TextGenerationModel(config.batch_size, config.seq_length,dataset.vocab_size).to(device)
        
        if use_cuda:
            model.load_state_dict(torch.load("model.pt"))
        else :
            trained_model = torch.load("model.pt", map_location=lambda storage, loc: storage)
            model.load_state_dict(trained_model)
        print(model)
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # fixme
    optimizer = optim.RMSprop(model.parameters(),config.learning_rate)  # fixme
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        
        # Only for time measurement of step through network
        t1 = time.time()
        
        #######################################################
        # Add more code here ...
        #######################################################
        
        model_output = model.forward(batch_inputs,use_cuda,config.temp)
        
        out_max = torch.argmax(model_output,dim=2)

        batch_targets = torch.stack(batch_targets)
        
        optimizer.zero_grad() 
        
        accuracy = 0.0

        model_output = model_output.view(-1,model_output.shape[2])

        batch_targets = batch_targets.view(-1).type(torch.LongTensor).type(dtype)
        
        loss = criterion(model_output,batch_targets)
        
        accuracy = accuracy_(model_output,batch_targets)
        
        loss.backward()
        
        optimizer.step()
        
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    int(config.train_steps), config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        text = torch.stack(batch_inputs)
        sentece1 = text[:,0].view(text[:,0].shape[0],1) 
#        print(dataset.convert_to_string(sentece1))

        if step % config.sample_every == 0:
            # Generate some sentences by sampling from the model
            text = torch.stack(batch_inputs)
            if generate_text:
                
                sentece = text[:,0].view(text[:,0].shape[0],1)            
                text = model.create_text(sentece,generated_text_size,use_cuda,config.temp)
                print("Generated Text : ",dataset.convert_to_string(text)," : end")
                
            if fixed_output_samples :

                sentece1 = text[:,0].view(text[:,0].shape[0],1) 
                gen_sentence1 = out_max[:,0].view(out_max[:,0].shape[0],1) 
                print("Original Text : ",dataset.convert_to_string(sentece1)," Generated Text : ",dataset.convert_to_string(gen_sentence1))
                sentece2 = text[:,1].view(text[:,1].shape[0],1) 
                gen_sentence2 = out_max[:,1].view(out_max[:,1].shape[0],1) 
                print("Original Text : ",dataset.convert_to_string(sentece2)," Generated Text : ",dataset.convert_to_string(gen_sentence2))
                sentece3 = text[:,2].view(text[:,2].shape[0],1) 
                gen_sentence3 = out_max[:,2].view(out_max[:,2].shape[0],1)
                print("Original Text : ",dataset.convert_to_string(sentece3)," Generated Text : ",dataset.convert_to_string(gen_sentence3))
            
            if fixed_random_samples :
                
                text = model.random_sampling(config.seq_length,use_cuda,config.temp)
                print("Generated Text : ",dataset.convert_to_string(text)," : end")
                
            
            print("Saving model...")
            torch.save(model.state_dict(), "model.pt")
    
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')

def list_to_numpy(list_t,config):
   
    targets = torch.zeros(config.seq_length,config.batch_size)
    for t,array in enumerate(list_t):
        targets[t,:]=array    
    
    return targets

def accuracy_(pred_out,true_out):
    
    predictions = torch.argmax(pred_out,dim=1)
    return (torch.sum(predictions == true_out).item())/true_out.shape[0]

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=200, help='How often to sample from the model')
    
    # additional
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--load_model', type=str, default="none", help="Load pretrained model or not")
    parser.add_argument('--temp', type=int, default=1, help="Temperature value")
    
    config = parser.parse_args()

    # Train the model
    train(config)
