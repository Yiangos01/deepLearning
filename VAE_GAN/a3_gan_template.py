import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from tensorboardX import SummaryWriter 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(args.latent_dim,128)
        self.linear2 = nn.Linear(128,256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256,512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.linear4 = nn.Linear(512,1024)
        self.batch_norm3 = nn.BatchNorm1d(1024)
        self.linear5 = nn.Linear(1024,784)
        self.LeakyRelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()
        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity

    def forward(self, z):
        # Generate images from z
        
        out = self.LeakyRelu(self.linear1(z))
        out = self.linear2(out)
        out = self.LeakyRelu(self.batch_norm1(out))
        out = self.linear3(out)
        out = self.LeakyRelu(self.batch_norm2(out))
        out = self.linear4(out)
        out = self.LeakyRelu(self.batch_norm3(out))
        out = self.tanh(self.linear5(out))
        
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(784,512)
        self.linear2 = nn.Linear(512,256)
        self.linear3 = nn.Linear(256,1)
        self.LeakyRelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

    def forward(self, img):
        # return discriminator score for img
        out = self.LeakyRelu(self.linear1(img))
        out = self.LeakyRelu(self.linear2(out))
        out = self.sigmoid(self.linear3(out))
        
        return out

def show(img,filename):
    npimg = img.detach().numpy()

    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
#    plt.imshow(npimg)
    plt.savefig(filename)
    
   
def gaussian_noice(z_dim, batch_size):

    epsilon = torch.randn((batch_size,z_dim))
                
    return epsilon
def train(dataloader, discriminator, generator, optimizer_G, optimizer_D,dtype,use_cuda):
    
#    writer = SummaryWriter('GAN_123')
    step=0
    for epoch in range(args.n_epochs):
        print("Epoch : ",epoch)
        for i, (imgs, _) in enumerate(dataloader):
            if use_cuda:    
                imgs.cuda()

            # Train Generator
            # ---------------
             
            noise = gaussian_noice(args.latent_dim, args.batch_size)
            
            fake_imgs = generator.forward(noise.type(dtype))
            
#            sample_images = fake_imgs.view(fake_imgs.shape[0],1,28,28)
#            show(make_grid(sample_images,nrow=16),"generator")
            fake_imgs_results = discriminator.forward(fake_imgs.type(dtype))
#            print(fake_imgs_results,torch.ones((args.batch_size,1)))
            generator_loss = torch.sum(-0.5 * torch.log(fake_imgs_results+1e-8))
            
            optimizer_G.zero_grad()
            
            generator_loss.backward()
        
            optimizer_G.step()
            
            # Train Discriminator
            # -------------------
            
            noise = gaussian_noice(args.latent_dim, args.batch_size)
            
            fake_imgs = generator.forward(noise.type(dtype))
            
            fake_imgs_results = discriminator.forward(fake_imgs.type(dtype))
            
            loss_for_fake_imgs = torch.sum(-0.5 * torch.log(1-(fake_imgs_results+1e-8)))
#            loss_for_fake_imgs = torch.sum(-0.5 * -torch.log(fake_imgs_results+1e-8))
            
            imgs = imgs.view(imgs.shape[0], imgs.shape[2]*imgs.shape[3])
            
            real_imgs_results = discriminator.forward(imgs.type(dtype))
            
            loss_for_real_imgs = torch.sum(-0.5 * torch.log(real_imgs_results+1e-8))

            discriminator_loss = loss_for_real_imgs + loss_for_fake_imgs
            
            optimizer_D.zero_grad()
            
            discriminator_loss.backward()
            
            optimizer_D.step()# Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            
#            writer.add_scalar('Generator loss',generator_loss,step)
#            writer.add_scalar('Discriminator loss',discriminator_loss,step)
            
            
            
            step+=1
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                print("Discriminator Loss : "+str(discriminator_loss)+" Generator Loss : "+str(generator_loss))
                gen_imgs = fake_imgs.view(fake_imgs.shape[0],1,28,28)
                save_image(gen_imgs[:25],'images/{}.png'.format(batches_done),nrow=5, normalize=True)
        
        torch.save(generator.state_dict(), "mnist_generator.pt")
                
            
def create_interpolation(generator,args,dtype,batches_done,sample_number):
    
    start_noise = gaussian_noice(args.latent_dim, 1)
    end_noise = gaussian_noice(args.latent_dim, 1)
    
    interpolate_noise = []
    for i in range(start_noise.shape[1]):
        interpolate_noise.append(torch.linspace(start=float(start_noise[0][i].detach().numpy()), end=float(end_noise[0][i].detach().numpy()), steps=9))

    interpolate_noise = torch.stack(interpolate_noise,dim=1)
    interpolate_imgs = generator.forward(interpolate_noise.type(dtype))
    interpolate_imgs = interpolate_imgs.view(interpolate_imgs.shape[0],1,28,28)
    save_image(interpolate_imgs,'interpolation/{}_{}.png'.format(batches_done,sample_number),nrow=9, normalize=True)

def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Running on GPU')
        dtype = torch.cuda.FloatTensor
    else :
        print('Running on cpu')
        dtype = torch.FloatTensor
    device = torch.device('cuda' if use_cuda else 'cpu')
    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D,dtype,use_cuda)


    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    for i in range(1000):
        create_interpolation(generator,args,dtype,i,0)
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--load', type=int, default=500,
                        help='load previous model')
    args = parser.parse_args()

    main()
