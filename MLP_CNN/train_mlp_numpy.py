"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
from tensorboardX import SummaryWriter 

writer = SummaryWriter('numpy_mlp/100_ADAM_lr0.0006')
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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
# =============================================================================
#   print("prediction shape : "+ str(predictions.shape))
#   print("target shape : " +str(targets.shape))
#   print(np.argmax(predictions, axis = 1))
#   print(np.argmax(targets, axis = 1))
#   print(targets)
# =============================================================================
  accuracy = np.sum(np.argmax(predictions, axis = 1) == np.argmax(targets, axis = 1)) / targets.shape[0]

  

  
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def softmax_n(predictions):
    """
    Performs a softmax transformation over logits. Maximum normalization is used for numerical stability (equivalent to log-sum-exp)
    :param logits: output of final (hidden) layer [n_classes, batch_size]
    :return: class probabilities [n_classes, batch_size]
    """    
    # subtract maximum logit per mini-batch for numerical stability
    
    exp_prob = np.exp(predictions-np.max(predictions, axis=1, keepdims=True))
    sum_exp_probs = exp_prob.sum(axis=1, keepdims=True)
    probs = exp_prob / sum_exp_probs
    
    
    return probs

def cross_entropy_loss(predictions,targets):
    
    """
    Args:
      logits: 2D float array of size [batch_size, self.n_classes].
                   The predictions returned through self.inference.
      labels: 2D int array of size [batch_size, self.n_classes]
                   with one-hot encoding. Ground truth labels for each
                   sample in the batch.
    Returns:
      loss: scalar float, full loss = cross_entropy + reg_loss
    """
    
    print("predictions shape : "+str(predictions.shape))
    print("targets shape : "+str(targets.shape))
    
    true_class = np.where(targets==1)
    
    print(true_class)
    
    loss = -np.sum(np.log(predictions[true_class])) / targets.shape[0]
    dout = (predictions - targets) / targets.shape[0];
    print(targets)
    print(predictions)
    print(dout)
    return loss,np.transpose(dout)

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
  
  hidden_layers = "100"
  
  if FLAGS.dnn_hidden_units:
    #dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = hidden_layers.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  print(dnn_hidden_units)

  
  cifar10 = cifar10_utils.get_cifar10("cifar10/cifar-10-batches-py")
#  lr = FLAGS.learning_rate
  MLP_net = MLP(n_inputs=FLAGS.batch_size,n_hidden=dnn_hidden_units,n_classes=10,lr=FLAGS.learning_rate)
  
  print(MLP_net)
  
  for step in range(FLAGS.max_steps):
      
      # Get batch and reshape input to vector
      x_train, y_train = cifar10['train'].next_batch(FLAGS.batch_size)
      x_train = np.reshape(x_train, (FLAGS.batch_size,-1))
      
      # Forward pass
      net_output = MLP_net.forward(x_train,y_train)
      
#      print("output shape : " + str(net_output.shape) )
#      print(np.sum(np.transpose(net_output)[1]))
      
      loss = MLP_net.net[len(MLP_net.net)-1].forward(np.transpose(net_output),y_train)
      dout = MLP_net.net[len(MLP_net.net)-1].backward(np.transpose(net_output),y_train)
#     
      batch_accuracy = accuracy(np.transpose(net_output),y_train)
      
#      print("lsajdkaskd"+str(dout.shape)+str(y_train.shape))
      MLP_net.backward(np.transpose(dout))
      
      if (step+1)%FLAGS.eval_freq==0:
#          print("in test")
          x_test , y_test = cifar10['test'].images, cifar10['test'].labels
#          x_test = batch_norm(torch.from_numpy(x_test)).detach().numpy()
          x_test = np.reshape(x_test, (x_test.shape[0],-1))
          net_test_output = MLP_net.forward(x_test,y_test)
          print("test set accuracy for step "+str(step+1)+" : "+str(accuracy(np.transpose(net_test_output),y_test)))
          writer.add_scalar('Test_accuracy',accuracy(np.transpose(net_test_output),y_test),step)
#      print(batch_accuracy)
      writer.add_scalar('Train_accuracy',batch_accuracy,step)
      writer.add_scalar('Train_loss',loss,step)
      # Apply softmax
#      softmax_output = np.transpose(softmax_n(np.transpose(net_output)))
      
      # Batch accuracy
      
     
#      loss, dout = cross_entropy_loss(np.transpose(softmax_output),y_train)
    
#      print("dout shape : " + str(dout.shape))
      
#      MLP_net.backward(dout)
      
      
      
  
  
  ########################
  # PUT YOUR CODE HERE  #
  #######################
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