import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torch.distributions.normal as Norm
import torch.distributions.kl as KL

import matplotlib.pyplot as plt
import numpy as np

from model import VRNN
from utils import set_all_seeds, load_dataset, EarlyStopping
from config import Config
from losses import Losses

# set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

loss_types = Losses(device)

def loss_function(dist_params, x):
    
    encoder_mean, encoder_var, prior_mean, prior_var, decoder_mean, decoder_var = dist_params
    loss = 0.
    KL_loss_=0.
    recons_loss_=0.
    
    timesteps = x.size(1)
    
    for t in range(timesteps):
        
        # KL loss-------------------------------------------------------------------------------
        
        KL_loss = loss_types.kl_type_1(encoder_mean[t], encoder_var[t], prior_mean[t], prior_var[t])
        #KL_loss = losses.kl_type_2(encoder_mean[t], encoder_var[t], prior_mean[t], prior_var[t])
        #KL_loss = losses.kl_type_3(encoder_mean[t], encoder_var[t], prior_mean[t], prior_var[t])
        
        # reconstruction loss-------------------------------------------------------------------
        
        recons_loss = loss_types.recons_type_1(x[:, t, :], decoder_mean[t])
        #recons_loss = losses.recons_type_2(x[:, t, :], decoder_mean[t])
        #recons_loss = losses.recons_type_3(x[:, t, :], decoder_mean[t], decoder_var[t])
        
        #------------------------------------------------------------------------------------------
        loss += recons_loss + KL_loss
        KL_loss_+=KL_loss
        recons_loss_+=recons_loss
    
    return loss, KL_loss_, recons_loss_


def train(model, train_dataloader, conf, epoch, device):
   
    model.train()
    size = len(train_dataloader.dataset)
    train_loss = 0
    
    for batch_idx, (x, _) in enumerate(train_dataloader):
        x = x.to(device)
        x = x.squeeze()   # x is of dimensions (batch_size, 1, 28, 28) --> after squeezing --> (batch_size, 28, 28)
        #x /= 255
        x = (x - x.min().item()) / (x.max().item() - x.min().item())
        
        #----------------forward----------------------
        dist_params = model(x)
        loss, kl_loss, recons_loss = loss_function(dist_params, x)
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
            print(f"loss: {loss/conf.batch_size:>7f}  [{current:>5d}/{size:>5d}]") 
            print(f"KL_loss: {kl_loss.item()/conf.batch_size:.6f}, recons_loss: {recons_loss.item()/conf.batch_size:.6f}")
            
            # generating samples
            sample = model.sample(conf.x_dim, device)
            plt.imshow(sample.cpu().detach().numpy())
            plt.pause(1e-6)
            
        #break
    
    avg_train_loss = train_loss/size  # average training loss per epoch
    print('====> Average Train loss: {:.4f}'.format(avg_train_loss))
    
    return avg_train_loss


def test(model, test_dataloader, conf, epoch, device):
    
    model.eval()
    size = len(test_dataloader.dataset)
    test_loss = 0
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_dataloader):
            x = x.to(device)
            x = x.squeeze()
            #x /= 255
            x = (x - x.min().item()) / (x.max().item() - x.min().item())
        
            #-------forward----------------
            dist_params = model(x)
            loss,_,_ = loss_function(dist_params, x)
            
            '''
            if batch_idx % conf.print_every == 0:
                sample, z = model.sample(conf.x_dim, device)
                plt.imshow(sample.cpu().detach().numpy())
                plt.pause(1e-6)
            '''
            
            test_loss += loss.item()
    
    #--------logging-----------------
    avg_test_loss = test_loss/size  # average test loss per epoch
    print('====> Average Test loss: {:.4f}'.format(avg_test_loss))
    
    return avg_test_loss
        
    
    
def execute(model, train_dataoader, test_dataloader, conf, device):

    train_loss = []  # list of training losses for all epochs
    test_loss = []   # list of test losses for all epochs
    
    # instantiating the object of EarlyStopping class 
    early_stopping = EarlyStopping(patience=conf.patience, verbose=True)
    
    for ep in range(1, conf.n_epochs+1):
        
        print(f"Epoch {ep}\n-------------------------------")
        train_loss_per_epoch = train(model, train_dataloader, conf, ep, device)
        
        #break
        
        test_loss_per_epoch = test(model, test_dataloader, conf, ep, device)
        
        train_loss.append(train_loss_per_epoch)
        test_loss.append(test_loss_per_epoch)
        
        # saving model without early stopping
        """
        if ep % conf.save_every == 1:
            fn = 'saves/vrnn_state_dict_'+str(ep)+'.pt'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)
        """
        
        # early_stopping needs the validation/test loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(test_loss_per_epoch, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(f'{model_save_path}/checkpoint2.pt'))
        
    print("Execution complete")
    
    return model, train_loss, test_loss


if __name__ == '__main__':

    # instantiate the config class and set up the seeds    
    conf = Config()
    set_all_seeds(123)

    # set up the data loaders
    train_dataloader, test_dataloader = load_dataset(conf.batch_size)

    # instantiate the model and set up the optimizer    
    model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim, conf.n_layers)
    #model= nn.DataParallel(model, device_ids = conf.device_ids)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = conf.learning_rate)

    # run model
    model, train_loss, test_loss = execute(model, train_dataloader, test_dataloader, conf, device)
        