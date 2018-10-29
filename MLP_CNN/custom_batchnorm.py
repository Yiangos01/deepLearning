import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
#    raise NotImplementedError
    
    self.n_neurons = n_neurons
    self.eps = eps
    
    self.gamma = nn.Parameter(data=torch.Tensor(n_neurons), requires_grad=True)
    self.beta = nn.Parameter(data=torch.Tensor(n_neurons),requires_grad=True)
    
#    self.gamma = nn.init.normal_(self.gamma,mean=0,std=0.0001)
    self.gamma == nn.init.constant_(self.gamma,1)
    self.beta = nn.init.constant_(self.beta,0)
#    mu, sigma = 0, 0.0001 # mean and standard deviation
    
    
# =============================================================================
#     self.gamma = torch.from_numpy(np.random.normal(mu, sigma, (n_neurons ,)))
#     self.beta = torch.from_numpy(np.zeros((n_neurons,)))
#     self.gamma = self.gamma.type(torch.LongTensor)
#     self.beta = self.beta.type(torch.LongTensor)
# =============================================================================
#    self.gamma.data.uniform_(1)
#    self.beta.data.uniform_(0)
    
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
#    print("input",input.shape)
#    print("parameters",self.

    mean = torch.sum(input,dim=0)/input.shape[0]
    
    variance = torch.sum(torch.pow(input-mean,2),dim=0)/input.shape[0]
    
    var_2 = torch.sqrt(variance+self.eps)
    
    normalize = (input-mean) / var_2
#    print(torch.sum(torch.pow(normalize,2),dim=0))
#    print(normalize)
#    normalize = normalize.type(torch.LongTensor)

    out = self.gamma*normalize + self.beta
   
#    self.assertLess(np.max(np.abs(out.var(dim=0).data.numpy() - 1)), 1e-1)#    print(out.shape)
#    print(self.beta)
#    print("output shape",out)
#    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
#    raise NotImplementedError
#    print("input shape",input.shape)
    #step1: calculate mean
#    print(input)
   
    mean = torch.sum(input,dim=0)/input.shape[0]

    xmu = input - mean
    
    var = torch.sum(torch.pow(xmu,2),dim=0)/input.shape[0]

    sqrtvar = torch.sqrt(var + eps)

    ivar = 1/sqrtvar

    xhat = xmu * ivar

    out = gamma * xhat + beta

    ctx.eps = eps
    
    ctx.save_for_backward(xhat,gamma,xmu,ivar,sqrtvar,var)

    ########################
    # END OF YOUR CODE    #
    #######################
    
    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    xhat,gamma,xmu,ivar,sqrtvar,var = ctx.saved_tensors
    
    grad_beta = torch.sum(grad_output, dim=0)

    grad_gamma = torch.sum(grad_output*xhat, dim=0)
    dxhat = grad_output * gamma

    divar = torch.sum(dxhat*xmu, dim=0)
    dxmu1 = dxhat * ivar

    dsqrtvar = -1 /(sqrtvar**2) * divar
    
    dvar = 0.5 * (1 /np.sqrt(var+ctx.eps)) * dsqrtvar

    dsq = 1 /grad_output.shape[0]  * dvar

    dxmu2 = 2 * xmu * dsq

    dx1 = (dxmu1 + dxmu2)


    dx2 = 1. /grad_output.shape[0] * (-1 * torch.sum(dxmu1+dxmu2, dim=0))

    grad_input = dx1 + dx2

    ########################
    # END OF YOUR CODE    #
    #######################

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
#    raise NotImplementedError
    
    self.n_neurons = n_neurons
    self.eps = eps
    
    self.gamma = nn.Parameter(data=torch.Tensor(n_neurons), requires_grad=True)
    self.beta = nn.Parameter(data=torch.Tensor(n_neurons),requires_grad=True)
    
#    self.gamma = nn.init.normal_(self.gamma,mean=0,std=0.0001)
    self.gamma == nn.init.constant_(self.gamma,1)
    self.beta = nn.init.constant_(self.beta,0)
    
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
#    raise NotImplementedError
    batch_norm = CustomBatchNormManualFunction.apply
    out = batch_norm(input,self.gamma,self.beta)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
