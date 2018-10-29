import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        
        self.h = nn.Linear(28*28,hidden_dim)
        self.mu = nn.Linear(hidden_dim,z_dim)
        self.log_sigma = nn.Linear(hidden_dim,z_dim)
        self.tanh = nn.Tanh()
        self.z_dim = z_dim

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        
        h = self.tanh(self.h(input))
        mean = self.mu(h)
        log_var = self.log_sigma(h)
        
        return mean, log_var


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.linear1 = nn.Linear(z_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,28*28)
        self.tahn = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        """
        Perform forward pass of decoder.

        Returns mean with shape [batch_size, 784].
        """
        mean = None
        h = self.tahn(self.linear1(input))
        mean = self.sigmoid(self.linear2(h))

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
                
        average_negative_elbo = None

        mean,log_var = self.encoder.forward(input)
        
        var = torch.exp(log_var)
        
        sigma = torch.sqrt(var)
        
        kl_error = 0.5*torch.sum((1+log_var-torch.pow(mean,2)-var),dim=1)
        
        epsilon = self.gaussian_noice(input.shape[0])
        
        z = mean + epsilon * sigma
        
        y = self.decoder.forward(z)
        
        bernoulli_likelihood = input*torch.log(y) + (1-input)*torch.log((1-y))
        
        recon_Loss = torch.sum(bernoulli_likelihood,dim=1)
        
        elbo = recon_Loss + kl_error
        
        average_negative_elbo = -torch.mean(elbo)
        
        
#        print(average_negative_elbo)
        
        return average_negative_elbo

    def gaussian_noice(self, batch_size):

        epsilon = torch.randn((batch_size,self.z_dim))
                
        return epsilon
    
    def bernoulli(self, batch_size):

        epsilon = torch.randn((batch_size,self.z_dim))
                
        return epsilon

    
    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        
        z = torch.randn((n_samples,self.z_dim))
        
        y = self.decoder.forward(z)
                
        sample_images = y.view(n_samples,1,28,28)    
        
        sampled_ims, im_means = sample_images, y

        return sampled_ims, im_means

def show(img,filename):
    npimg = img.detach().numpy()

    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
#    plt.imshow(npimg)
    plt.savefig(filename)
    

def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = None
    list_elbos = []
    for im in data:
        x = im.view(im.shape[0], im.shape[2]*im.shape[3])
        average_negative_elbo = model.forward(x)
        list_elbos.append(average_negative_elbo)
                
        optimizer.zero_grad()
      
        average_negative_elbo.backward()
      
        optimizer.step()

    average_epoch_elbo = torch.mean(torch.stack(list_elbos))
#    raise NotImplementedError()

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)
    
    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)

def save_maniforld_plot(model,dim):
    
    values = torch.linspace(-2.5, 2.5, steps=dim)
    
    z_tensor = torch.zeros((20*20,2))
    print(z_tensor.shape)
    counter=0
    for i in range(len(values)):
        for j in range(len(values)):
             
            z = torch.Tensor([values[i],values[j]])
            z_tensor[counter,:]=z
            counter+=1
    
    y = model.decoder.forward(z_tensor)
    print(y.shape)
    sample_images = y.view(y.shape[0],1,28,28)   
    
    show(make_grid(sample_images,nrow=20),"manifold")
    print(sample_images.shape)
#   
    

def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    sampled_ims, im_means = model.sample(20)
    show(make_grid(sampled_ims,nrow=5),"samples_0")
    for epoch in range(ARGS.epochs):
        print(epoch)
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch "+str(epoch)+"] train elbo: "+str(train_elbo.detach().numpy())+" val_elbo: "+str(val_elbo.detach().numpy()))

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if epoch%10==0 and epoch!=0:
            sampled_ims, im_means = model.sample(20)
            show(make_grid(sampled_ims,nrow=5),"samples_"+str(epoch))
        
    sampled_ims, im_means = model.sample(20)
    show(make_grid(sampled_ims,nrow=5),"samples_end")
    
    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        save_maniforld_plot(model,20)
    
    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
