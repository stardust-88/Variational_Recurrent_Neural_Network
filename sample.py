import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import VRNN
from config import Config


state_dict = torch.load('saves/--file_name--')
model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim, conf.n_layers)
model.load_state_dict(state_dict)

sample = model.sample(28*6)
plt.imshow(sample.numpy(), cmap='gray')
plt.show()