"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
from tensorboardX import SummaryWriter 

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

writer = SummaryWriter('pytorch_mlp/300x3_ADAM_lr0.0006')
# Directory in which cifar data is saved
DATA_DIR_DEFAULT = "cifar10/cifar-10-batches-py"

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
  accuracy = np.sum(np.argmax(predictions, axis = 1) == np.argmax(targets, axis = 1)) / targets.shape[0]
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  #dnn_hidden_units = [200,200]
  
  #batch_size = 200
  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  
  x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
#  print(x_train.shape)
  
  MLP_net = MLP(n_inputs=1*3*32*32,n_hidden=dnn_hidden_units,n_classes=10)
  
  params = MLP_net.parameters()
  criterion = torch.nn.CrossEntropyLoss()
#  criterion = torch.nn.L1Loss()
#  optimizer = torch.optim.SGD(params,lr=FLAGS.learning_rate)#,momentum=0.005)# weight_decay=0.001)
  optimizer = torch.optim.Adam(params,lr=FLAGS.learning_rate)#,weight_decay=0.0001)
#  optimizer = torch.optim.SGD(params,lr=0.02)
#  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.8)
  print(MLP_net)
  
  batch_norm = torch.nn.BatchNorm2d(3)#,affine=False,momentum=0)
  
  loss_list = []
  for step in range(FLAGS.max_steps):  
      # Get batch and reshape input to vector
      x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
      
      x_train = batch_norm(torch.from_numpy(x_train)).detach().numpy()
           
      x_train = np.reshape(x_train, (FLAGS.batch_size,-1))
      
      net_output = MLP_net.forward(torch.from_numpy(x_train))
      
      batch_accuracy = accuracy(net_output.detach().numpy(),y_train)
      
      y_train = torch.from_numpy(y_train)
      y_train = y_train.type(torch.LongTensor)
#      y_train = y_train.type(torch.FloatTensor)

      loss = criterion(net_output,torch.max(y_train, 1)[1])
      loss_list.append(loss)
      #      print("loss : ",loss)
      
      optimizer.zero_grad()
      
      loss.backward()
      
      optimizer.step()
      
#      scheduler.step()
#      print("out and y shapes : "+str(net_output.shape),str(y_train.shape))
      if (step+1)%FLAGS.eval_freq==0:
#          print("in test")
          x_test , y_test = cifar10['test'].images, cifar10['test'].labels
          x_test = batch_norm(torch.from_numpy(x_test)).detach().numpy()
          x_test = np.reshape(x_test, (x_test.shape[0],-1))
          net_test_output = MLP_net.forward(torch.from_numpy(x_test))
          print("test set accuracy for step "+str(step+1)+" : "+str(accuracy(net_test_output.detach().numpy(),y_test)))
          print("loss : ",sum(loss_list)/len(loss_list))
          loss_list = []
          writer.add_scalar('Test_accuracy',accuracy(net_test_output.detach().numpy(),y_test),step)

      writer.add_scalar('Train_accuracy',batch_accuracy,step)
      writer.add_scalar('Train_loss',loss,step)
#  raise NotImplementedError
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
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
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