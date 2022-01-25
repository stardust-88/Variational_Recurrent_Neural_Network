# configuration class for hyperparameters and other small things
import os
class Config(object):
    
    def __init__(self):
        
        # hyperparameters
        self.x_dim = 28
        self.h_dim = 100
        self.z_dim = 16
        self.n_layers = 1
        self.n_epochs = 100
        self.clip = 10
        self.batch_size = 128
        self.learning_rate = 0.001
        self.patience = 5
        
        # other
        self.print_every = 100
        self.save_every = 10
        self.seed = 123
        self.device_ids = [0,1,2,3]
        
        # paths
        self.path = os.getcwd()[:-4]
        self.model_save_path = f'{self.path}/models'