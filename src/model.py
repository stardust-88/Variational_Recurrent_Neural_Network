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
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU()
        )
        
        # extracting features of the the latent variable z<t>
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU()
        )
        
        # encoder
        self.encoder = nn.Sequential(
            #nn.Linear(h_dim + h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            nn.Linear(2*h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        
        self.encoder_mean = nn.Sequential(
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        
        self.encoder_var = nn.Sequential(
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )
        
        # prior distribution and its parameters
        self.prior = nn.Sequential(
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        
        self.prior_mean = nn.Sequential(
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            nn.Linear(h_dim, z_dim)
        )
        
        self.prior_var = nn.Sequential(
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.Softplus()
        )
        
        # decoder
        self.decoder = nn.Sequential(
            #nn.Linear(h_dim + h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            nn.Linear(2*h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        
        self.decoder_mean = nn.Sequential(
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )
        self.decoder_var = nn.Sequential(
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            #nn.Linear(h_dim, h_dim),
            #nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Softplus()
        )
        
        # recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers)
        #self.rnn = nn.LSTM(h_dim + h_dim, h_dim, n_layers)
        
        
    def inference(self, phi_x_t, ht_minus_1):
        
        # returns the parameters of the posterior distribution
        
        encoder_input = torch.cat([phi_x_t, ht_minus_1], dim=1)
        encoder_t = self.encoder(encoder_input)
        encoder_mean_t = self.encoder_mean(encoder_t)
        encoder_var_t = self.encoder_var(encoder_t)
        
        return encoder_mean_t, encoder_var_t
    
    
    def generation_z(self, ht_minus_1):
        
        # returns the parameters of the prior distribution
        
        prior_t = self.prior(ht_minus_1)
        prior_mean_t = self.prior_mean(prior_t)
        prior_var_t = self.prior_var(prior_t)
        
        return prior_mean_t, prior_var_t
    
    
    def generation_x(self, phi_z_t, ht_minus_1):
        
        # returns the parameters of the output distribution
        
        decoder_input = torch.cat([phi_z_t, ht_minus_1], dim=1)
        decoder_t = self.decoder(decoder_input)
        decoder_mean_t = self.decoder_mean(decoder_t)
        decoder_var_t = self.decoder_var(decoder_t)
        
        return decoder_mean_t, decoder_var_t
    
    
    def recurrence(self, phi_x_t, phi_z_t, h, c=0):
        
        rnn_input = torch.cat([phi_x_t, phi_z_t], dim=1).unsqueeze(0)
        _, h = self.rnn(rnn_input, h) # gru
        #_, (h,c) = self.rnn(rnn_input, (h,c)) # lstm
        
        return h
    
    
    def reparameterize(self, *args):
        z_mean, z_log_var = args
        
        # sampling from a standard normal distribution
        #eps = torch.randn(z_mean.size(0), z_mean.size(1)).to(device)
        
        # creating a random variable z drawn from a normal distribution having parameters z_mu and z_log_var
        #z = z_mean + eps*torch.exp(z_log_var/2.)

        eps = torch.FloatTensor(z_log_var.size()).normal_().to(device)
        return eps.mul(z_log_var).add_(z_mean)
        #return z

    
    
    def forward(self, x):
        
        all_encoder_mean, all_encoder_var = [], []
        all_prior_mean, all_prior_var = [], []
        all_decoder_mean, all_decoder_var = [], []
        
        timesteps = x.size(1) # timesteps = 28  (x is of shape = (batch_size, 28, 28))
        h = torch.zeros([self.n_layers, x.size(0), self.h_dim], device = x.device) # no_of_layers x batch_size x h_dim
        #c = torch.zeros([self.n_layers, x.size(0), self.h_dim], device = x.device) # cell state if using lstm
        
        for t in range(timesteps):
            
            # feature extraction for x_t
            phi_x_t = self.phi_x(x[:, t, :])   # x[:, t, :] has dimensions (batch_size, 28)
        
            # encoder
            encoder_mean_t, encoder_var_t = self.inference(phi_x_t, h[-1])
            
            # reparameterization
            z_t = self.reparameterize(encoder_mean_t, encoder_var_t)
            
            # feature extraction for z_t
            phi_z_t = self.phi_z(z_t)
            
            # decoder
            decoder_mean_t, decoder_var_t = self.generation_x(phi_z_t, h[-1])
            #print("decoder mean vector size: ",decoder_mean_t.size())  # (batch_size, 28)
                
            # prior
            prior_mean_t, prior_var_t = self.generation_z(h[-1]) # gru
            
            # recurrence
            h = self.recurrence(phi_x_t, phi_z_t, h)
            
            all_encoder_mean.append(encoder_mean_t)
            all_encoder_var.append(encoder_var_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_var.append(prior_var_t)
            all_decoder_mean.append(decoder_mean_t)
            all_decoder_var.append(decoder_var_t)
            
        return [all_encoder_mean, all_encoder_var, all_prior_mean, all_prior_var, all_decoder_mean, all_decoder_var]
    
    
    # To sample from the prior distribution
    def sample(self, seq_len, device, get_latent_vector = False):
        
        sample = torch.zeros(seq_len, self.x_dim, device = device)
        h = torch.zeros(self.n_layers, 1, self.h_dim, device = device)
        #c = torch.zeros(self.n_layers, 1, self.h_dim, device = device) # cell state if using lstm
        
        if get_latent_vector == True:
            z = torch.zeros(seq_len, self.z_dim, device = device)
        
        for t in range(seq_len):
            
            # prior
            prior_mean_t, prior_var_t = self.generation_z(h[-1])
            
            # reparameterization
            z_t = self.reparameterize(prior_mean_t, prior_var_t)
             
            # stacking the latent vectors into a matrix 
            if get_latent_vector == True:
                z[t] = z_t.data
            
            # feature extraction for z_t
            phi_z_t = self.phi_z(z_t)
            
            # decoder
            decoder_mean_t, _ = self.generation_x(phi_z_t, h[-1])
            
            # sampling the x_t (the reconstructed output)
            #x_t = self.reparameterize(decoder_mean_t, decoder_var_t)
            
            #phi_x_t = self.phi_x(x_t)
            phi_x_t = self.phi_x(decoder_mean_t)
            
            # recurrence
            h = self.recurrence(phi_x_t, phi_z_t, h) # gru
            
            #print(x_t.data)
            sample[t] = decoder_mean_t.data
            #sample[t] = x_t.data
            
        if get_latent_vector == True:
            return sample, z
        return sample
    
 
    # To encode the th given input into it's latent representation
    def encode(self, x):
        
        timesteps = x.size(1)  # timesteps = 28  (x is of shape = (28, 28))
        h = torch.zeros([self.n_layers, 1, self.h_dim], device = x.device) # no_of_layers x batch_size x h_dim
        
        z = torch.zeros(timesteps, self.z_dim, device = x.device)
        
        for t in range(timesteps):    
            
            # feature extraction for x_t
            phi_x_t = self.phi_x(x[t].unsqueeze(0))  # x is (28,28), x[t] is (28,), x[t].unsqueeze(0) is (1,28)
            
            # encoder
            encoder_mean_t, encoder_var_t = self.inference(phi_x_t, h[-1])
            
            # reparameterization
            z_t = self.reparameterize(encoder_mean_t, encoder_var_t)
            
            z[t] = z_t.data
            
            # feature extraction for z_t
            phi_z_t = self.phi_z(z_t)
            
            # recurrence
            h = self.recurrence(phi_x_t, phi_z_t, h) # gru
            
        return z
    
    
    # for reconstructing the output from the given input by samping latent vector from posterior distribution
    def reconstruction_from_posterior(self, x):
        
        timesteps = x.size(1)  # timesteps = 28  (x is of shape = (28, 28))
        h = torch.zeros([self.n_layers, 1, self.h_dim], device = x.device) # no_of_layers x batch_size x h_dim
        
        seq_len = x.size(1)
        sample_post = torch.zeros(seq_len, self.x_dim, device = x.device)
        
        for t in range(timesteps):
            
            # feature extraction for x_t
            phi_x_t = self.phi_x(x[t].unsqueeze(0))  # x is (28,28), x[t] is (28,), x[t].unsqueeze(0) is (1,28)
            
            # encoder
            encoder_mean_t, encoder_var_t = self.inference(phi_x_t, h[-1])
            
            # reparameterization
            z_t = self.reparameterize(encoder_mean_t, encoder_var_t)
            
            # feature extraction for z_t
            phi_z_t = self.phi_z(z_t)
            
            # decoder
            decoder_mean_t, decoder_var_t = self.generation_x(phi_z_t, h[-1])
            
            # recurrence
            h = self.recurrence(phi_x_t, phi_z_t, h)
            
            sample_post[t] = decoder_mean_t.data
            
        return sample_post
            