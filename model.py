import torch
import torch.nn as nn
import numpy as np


class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers):
        super(VRNN, self).__init__()
        
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        
        # feature extractions
        # extracting features of the input x<t> 
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        
        # extracting features of the the latent variable z<t>
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        
        self.encoder_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        
        self.encoder_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )
        
        # prior distribution and its parameters
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        
        self.prior_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim)
        )
        
        self.prior_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        
        self.decoder_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )
        self.decoder_var = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Softplus()
        )
        
        # recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers)
        
    def forward(self, x):
        
        all_encoder_mean, all_encoder_var = [], []
        all_prior_mean, all_prior_var = [], []
        all_decoder_mean = []
        
        timesteps = x.size(1) # timesteps = 28  (x is of shape = (28, batch_size, 28))
        h = torch.zeros([self.n_layers, x.size(0), self.h_dim], device = x.device) # no_of_layers x batch_size x h_dim
        #c = torch.zeros([self.n_layers, x.size(0), self.h_dim], device = x.device) # cell state if using lstm
        
        for t in range(timesteps):
            
            # feature extraction for x_t
            phi_x_t = self.phi_x(x[:, t, :])
        
            # encoder
            encoder_t = self.encoder(torch.cat([phi_x_t, h[-1]], dim=1))
            encoder_mean_t = self.encoder_mean(encoder_t)
            encoder_var_t = self.encoder_var(encoder_t)
            
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_var_t = self.prior_var(prior_t)
            
            # reparameterization
            z_t = self.reparameterize(encoder_mean_t, encoder_var_t)
            
            # feature extraction for z_t
            phi_z_t = self.phi_z(z_t)
            
            # decoder
            decoder_t = self.decoder(torch.cat([phi_z_t, h[-1]], dim=1))
            decoder_mean_t = self.decoder_mean(decoder_t)
            decoder_var_t = self.decoder_var(decoder_t)
            
            # recurrence
            #_, (h, c) = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0), (h, c)) # lstm 
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0), h)  # gru
            
            all_encoder_mean.append(encoder_mean_t)
            all_encoder_var.append(encoder_var_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_var.append(prior_var_t)
            all_decoder_mean.append(decoder_mean_t)
            
        return [all_encoder_mean, all_encoder_var, all_prior_mean, all_prior_var, all_decoder_mean]
    
    
    def sample(self, seq_len, device):
        
        sample = torch.zeros(seq_len, self.x_dim, device = device)
        h = torch.zeros(self.n_layers, 1, self.h_dim, device = device)
        #c = torch.zeros(self.n_layers, 1, self.h_dim, device = device) # cell state if using lstm
        
        for t in range(seq_len):
            
            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_var_t = self.prior_var(prior_t)
            
            # reparameterization
            z_t = self.reparameterize(prior_mean_t, prior_var_t, device)
            
            z = z_t
            #print(z)
            
            # feature extraction for z_t
            phi_z_t = self.phi_z(z_t)
            
            # decoder
            decoder_t = self.decoder(torch.cat([phi_z_t, h[-1]], dim=1))
            decoder_mean_t = self.decoder_mean(decoder_t)
            
            phi_x_t = self.phi_x(decoder_mean_t)
            
            # recurrence
            #_, (h, c) = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0), (h, c)) # lstm 
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0), h)  # gru
            
            sample[t] = decoder_mean_t.data
            
        return sample, z
    
    
    def reparameterize(self, *args):
        z_mean, z_log_var, device = args
        
        # sampling from a standard normal distribution
        eps = torch.randn(z_mean.size(0), z_mean.size(1)).to(device)
        
        # creating a random variable z drawn from a normal distribution having parameters z_mu and z_log_var
        z = z_mean + eps*torch.exp(z_log_var/2.)
        return z