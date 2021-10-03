class Config(object):
    
    def __init__(self):
        
        # hyperparameters
        self.x_dim = 28
        self.h_dim = 100
        self.z_dim = 16
        self.n_layers = 1
        self.n_epochs = 1
        self.clip = 10
        self.batch_size = 512
        self.learning_rate = 0.001
        
        # other
        self.print_every = 100
        self.save_every = 10
        self.seed = 123
        self.device_ids = [0]