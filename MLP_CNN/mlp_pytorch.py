"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
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
    super(MLP, self).__init__()
    
    linear_list = [nn.Linear(n_inputs,n_hidden[0])]
    prev_layer_neurons = n_hidden[0]
    for next_layer_neurons in n_hidden[1:]:
        linear_list.append(nn.Linear(prev_layer_neurons,next_layer_neurons))
        prev_layer_neurons = next_layer_neurons
    linear_list.append(nn.Linear(prev_layer_neurons,n_classes))
    self.linears = nn.ModuleList(linear_list)
    
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()
# =============================================================================
#         name = "linear_"+str(i)
#         self.name = nn.Linear(prev_layer_neurons,next_layer_neurons)
#         prev_layer_neurons = next_layer_neurons
# =============================================================================
    
    self.output_layer = nn.Linear(prev_layer_neurons,n_classes)

  def forward(self, x):
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
#    print("In forward ...")
    
    for i, l in enumerate(self.linears):
        if i == len(self.linears)-1:
            x = self.linears[i](x)
        else :
            x = self.relu(self.linears[i](x))
#    print(x.shape)
    out=self.softmax(x)
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
