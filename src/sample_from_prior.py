import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import VRNN
from config import Config


state_dict = torch.load(f'{conf.model_save_path}/checkpoint2.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim, conf.n_layers)
model = model.to(device)
model.load_state_dict(state_dict)

# generating samples from prior distribution
sample = model.sample(28, device)
plt.imshow(sample.cpu().detach().numpy())
plt.show()