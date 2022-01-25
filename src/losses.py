import torch
import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch.nn.functional as F
import numpy as np

class Losses:
    def __init__(self, device):
        self.small = torch.tensor([1e-10]).to(device)
        
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        
        kld_element =  (2 * torch.log(torch.max(std_2,self.small)) - 2 * torch.log(torch.max(std_1,self.small)) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /(std_2.pow(2)+1e-10) - 1)
        
        return 0.5 * torch.sum(kld_element)    
    
    
    def KLD_loss(mean_1, std_1, mean_2, std_2):
        
        ret = -0.5 * torch.sum(std_1 - std_2 - torch.div(std_1.exp() + (mean_1 - mean_2).pow(2), std_2.exp()+1e-10))
        
        return ret
    
    
    def kl_type_1(self, mean_1, std_1, mean_2, std_2):
        
        KL_loss = self._kld_gauss(mean_1, std_1, mean_2, std_2)
        
        return KL_loss
    
    def kl_type_2(self, mean_1, std_1, mean_2, std_2):
        
        norm_dist_1 = Norm.Normal(mean_1, std_1)
        norm_dist_2 = Norm.Normal(mean_2, std_2)
        KL_loss = torch.mean(KL.kl_divergence(norm_dist_2, norm_dist_1))
        
        return KL_loss
    
    
    def kl_type_3(self, mean_1, std_1, mean_2, std_2):
        
        KL_loss = self.KLD_loss(mean_1, std_1, mean_2, std_2)
        
        return KL_loss
    
    
    def _nll_bernoulli(self, x, theta):
        return - torch.sum(x*torch.log(torch.max(theta, self.small)) + (1-x)*torch.log(torch.max(1-theta, self.small)))
    
    def Gaussian_nll(self, y, mu, sig):
        
        #nll = 0.5 * torch.sum(torch.square(y - mu) / sig**2 + 2 * torch.log(sig) + torch.log(torch.tensor(2 * np.pi)), axis=-1)
        nll = 0.5 * torch.sum(torch.square(y - mu) / (sig**2+1e-10) + 2 * torch.log(torch.max(sig,self.small)) + torch.log(torch.tensor(2 * np.pi)))
        
        return nll
    
    def recons_type_1(self, x, theta):
        
        recons_loss = _nll_bernoulli(x, theta)
        
        return recons_loss
    
    
    def recons_type_2(self, x, theta):
        
        recons_loss = torch.mean(F.binary_cross_entropy(theta, x, reduction = 'none'))
        
        return recons_loss
    
    
    def recons_type_3(self, y, mu, sig):
        
        recons_loss = Gaussian_nll(y, mu, sig)
        #recons_loss = torch.mean(recons_loss) # when axis axis=-1 in Gaussian_nll
        
        return recons_loss 