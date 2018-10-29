"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
#  raise NotImplementedError
  accuracy = np.sum(np.argmax(predictions, axis = 1) == np.argmax(targets, axis = 1)) / targets.shape[0]
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10("cifar10/cifar-10-batches-py")
  
#  x_train, y_train = cifar10['train'].next_batch(32)
#  print(x_train.shape)
  
  use_cuda = torch.cuda.is_available()
  if use_cuda:
      print('Running on GPU')
  device = torch.device('cuda' if use_cuda else 'cpu')
  dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
  
  conv_net = ConvNet(n_channels=3,n_classes=10).to(device)
  
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(conv_net.parameters(),lr=FLAGS.learning_rate)
      
  loss_list=[]
  for step in range(FLAGS.max_steps):  
        # Get batch and reshape input to vector
        x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
#        x_train = np.reshape(x_train, (batch_size,-1))
        print("ok")
        net_output = conv_net.forward(torch.from_numpy(x_train).type(dtype))
        
#        print(net_output.shape)
        
#        print("batch accuracy : "+str(accuracy(net_output.cpu().detach().numpy(),y_train)))
        
        y_train = torch.from_numpy(y_train).type(dtype)

        loss = criterion(net_output,torch.max(y_train, 1)[1])
        loss_list.append(loss.cpu().detach().numpy())
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        if (step+1)%FLAGS.eval_freq==0:
#         print("in test")
          net_test_output_list=[]
          for i in range(100):
              x_test , y_test = cifar10['test'].next_batch(100)
#              y_test = torch.from_numpy(y_test).type(dtype)
#              print(x_test.shape)
#          x_test = np.reshape(x_test, (x_test.shape[0],-1))
              net_test_output = conv_net.forward(torch.from_numpy(x_test).type(dtype))
              
              net_test_output_list.append(accuracy(net_test_output.cpu().detach().numpy(),y_test))
          print("test set accuracy for step "+str(step+1)+" : "+str(sum(net_test_output_list)/len(net_test_output_list)))
          print("loss : ",sum(loss_list)/len(loss_list))
#        break
   

  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
