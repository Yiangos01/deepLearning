"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    
    mu, sigma = 0, 0.0001 # mean and standard deviation
    
    self.params['weight'] = np.random.normal(mu, sigma, (in_features ,out_features))
    self.params['bias'] = np.zeros((out_features,))
    
    self.grads['weight'] = np.zeros((in_features, out_features))
    self.grads['bias'] = np.zeros((out_features,))
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
#    print("in Linear Forward")
    self.input = x
    pre_out = np.dot(x,self.params['weight'])
    out = pre_out + self.params['bias']
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
#    print("shapes linear"+str(dout.shape)+str(np.transpose(self.params['weight']).shape))
    dx = np.dot(dout,np.transpose(self.params['weight']))
    
    self.grads['weight'] = np.dot(np.transpose(self.input),dout)
    self.grads['bias'] =  np.sum(dout,axis=0)
      
    
#    self.params['weight'] -= 0.002 * self.grads['weight']
#    self.params['bias'] -= 0.002 * self.grads['bias']
    
    
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
#    print("In Relu forward")
    
    self.input = x
    x_threshold = x > 0
    out = x * x_threshold
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
# =============================================================================
#     x=self.input
#     x[x>0]=1
#     x[x<0]=0
#     
#     dx = x * dout
# =============================================================================
    dx = dout*(self.input>0).astype(int)
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    
    
    exp_prob = np.exp(x-np.max(x, axis=0, keepdims=True))
    sum_exp_probs = exp_prob.sum(axis=0, keepdims=True)
    out = exp_prob / sum_exp_probs
    self.probs = out
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
   
#    dx = dout
#    print("proba shape")
#    print(self.probs.shape)
    dx = np.zeros(dout.shape)
    
#    jacobian_list = []
    for n in range(dout.shape[1]):
        jacobian_m = np.zeros((dout.shape[0],dout.shape[0]))
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = self.probs[i][n] * (1-self.probs[j][n])
                else: 
                    jacobian_m[i][j] = -self.probs[i][n]*self.probs[j][n]
#        jacobian_list.append(jacobian_m)
        dx[:,n] = np.dot(jacobian_m,dout[:,n])
#    print(len(jacobian_list))
#    print(jacobian_list[0].shape,jacobian_list[1].shape)
    
#    for i,jacobian_m in enumerate(jacobian_list):
#        dx[:,i] = np.dot(jacobian_m,dout[:,i])
        

    
    
    
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODO:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
#    print(x.shape,y.shape)
    
#    print(x.shape)
#    print(y.shape)
    true_class = np.where(y==1)
#    print(true_class)
#    print(y.shape)
#    print(x.shape)
    out = - np.sum(np.log(x[true_class])) / y.shape[0]
#    print('loss'+str(out))
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODO:
    Implement backward pass of the module.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = np.zeros((y.shape))
    
    true_class = np.where(y==1)
    
    losses = (-1/x[true_class]) / y.shape[0]
    for i,j in enumerate(true_class[1]):
        dx[i,j] = losses[i] 
    ########################
    # END OF YOUR CODE    #
    #######################
#    print(dx.shape)
    return dx
