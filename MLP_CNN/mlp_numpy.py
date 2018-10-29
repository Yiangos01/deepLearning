"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

import numpy as np

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes,lr):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODO:
    Implement initialization of the network.
    """
    
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    # Initialize Weights and biases
    print("initialization of the network...")

#    n_modules = len(W_list)
    self.lr = lr
    self.net = []
    self.net.insert(0,CrossEntropyModule())
    self.net.insert(0,SoftMaxModule())
    
    
#    prev_layer_neurons = 3072
    next_layer_neurons = 10
    for  n_hidden_neurons in reversed(n_hidden):
        self.net.insert(0,LinearModule(n_hidden_neurons,next_layer_neurons))
        self.net.insert(0,ReLUModule())
        next_layer_neurons = n_hidden_neurons
        
    self.net.insert(0,LinearModule(3072,next_layer_neurons))
    
    print(self.net)
    print("network ready")
    

  
#    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x,y):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    x_input = x
    for i in range(len(self.net)-1):
#        print(x_input.shape)
        if isinstance(self.net[i],SoftMaxModule):
            x_input=np.transpose(x_input)
            out = self.net[i].forward(x_input)
        else:
            out = self.net[i].forward(x_input)
        #out = self.net[i].forward(x_input)
#        print(self.net[i])
#        print("module : "+str(i)+" output shape : "+str(out.shape))
        x_input = out
        
    
#    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    dout_new = dout
#    print("Start bachward pass ...")
#    print(dout_new)
    # Backward Pass
    for l in range(len(self.net)-2,-1,-1):
        
        dout_new = self.net[l].backward(dout_new)
        
        if isinstance(self.net[l],SoftMaxModule):
            dout_new = np.transpose(dout_new)
        
        if isinstance(self.net[l],LinearModule):
            self.net[l].params['weight'] -= self.lr * self.net[l].grads['weight']
            self.net[l].params['bias'] -= self.lr * self.net[l].grads['bias']
#        print(self.net[l])
#        print(dout_new.shape)
#        delta_new = np.dot(np.transpose(self.modules[l].W),delta) * self.modules[l-1].relu_der()
#        delta=delta_new
        
    ########################
    # END OF YOUR CODE    #
    #######################

    return
