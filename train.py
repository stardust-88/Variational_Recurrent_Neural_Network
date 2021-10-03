import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torch.distributions.normal as Norm
import torch.distributions.kl as KL

import matplotlib.pyplot as plt
import numpy as np
from model import VRNN
from utils import set_all_seeds, load_dataset
from config import Config

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_function(dist_params, x):
    
    encoder_mean, encoder_var, prior_mean, prior_var, decoded_x = dist_params
    loss = 0.
    
    for i in range(x.size(1)):
        
        # KL loss
        norm_dist_1 = Norm.Normal(prior_mean[i], prior_var[i])
        norm_dist_2 = Norm.Normal(encoder_mean[i], encoder_var[i])
        KL_loss = torch.mean(KL.kl_divergence(norm_dist_1, norm_dist_2))
        
        # reconstruction loss
        recons_loss = torch.mean(F.binary_cross_entropy(decoded_x[i], x[:, i, :], reduction = 'none'))
        
        loss += recons_loss + KL_loss
    
    return loss



def train(model, train_dataloader, conf, epoch):
   
    model.train()
    size = len(train_dataloader.dataset)
    train_loss = 0
    
    for batch_idx, (x, _) in enumerate(train_dataloader):
        x = x.to(device)
        x = x.squeeze()
        #x /= 255
        
        #----------------forward----------------------
        dist_params = model(x)
        loss = loss_function(dist_params, x)
        train_loss += loss.item()
        
        #------------backward------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _ = nn.utils.clip_grad_norm_(model.parameters(), conf.clip)
        
        # ----------logging---------------
        if batch_idx % conf.print_every == 0:
            current = batch_idx*len(x)
            loss = loss.item()
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") 
            
            # generating samples
            sample, _ = model.sample(conf.x_dim, device)
            plt.imshow(sample.cpu().detach().numpy())
            plt.pause(1e-6)
    
    avg_train_loss = train_loss/size
    print('====> Average Train loss: {:.4f}'.format(avg_train_loss))
    
    
    
def test(model, test_dataloader, conf, epoch):
    
    model.eval()
    size = len(test_dataloader.dataset)
    test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_dataloader):
            x = x.to(device)
            x = x.squeeze()
            #x /= 255
        
            #-------forward----------------
            dist_params = model(x)
            loss = loss_function(dist_params, x)
            
            """
            if batch_idx % conf.print_every == 0:
                sample, z = model.sample(conf.x_dim, device)
                plt.imshow(sample.cpu().detach().numpy())
                plt.pause(1e-6)
            """
            
            test_loss += loss.item()
    
    #--------logging-----------------
    avg_test_loss = test_loss/size
    print('====> Average Test loss: {:.4f}'.format(avg_test_loss))
        
        

def execute(model, train_dataoader, test_dataloader, conf):
    
    for ep in range(1, conf.n_epochs+1):
        print(f"Epoch {ep}\n-------------------------------")
        train(model, train_dataloader, conf, ep)
        test(model, test_dataloader, conf, ep)
        
    print("Execution complete")

    
if __name__ == '__main__':
    
    # instantiate the config class and set up the seeds    
    conf = Config()
    set_all_seeds(123)

    # set up the data loaders
    train_dataloader, test_dataloader = load_dataset(conf.batch_size)
  
    # instantiate the model and set up the optimizer    
    model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim, conf.n_layers)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = conf.learning_rate)
    
    # run model
    execute(model, train_dataloader, test_dataloader, conf)
        