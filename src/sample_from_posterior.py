import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import VRNN
from config import Config
from utils import load_dataset

state_dict = torch.load(f'{conf.model_save_path}/checkpoint2.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VRNN(conf.x_dim, conf.h_dim, conf.z_dim, conf.n_layers)
model = model.to(device)
model.load_state_dict(state_dict)

def get_data(data_loader):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    #print(images.size())  # ---> torch.Size([128, 1, 28, 28]) # batch_size number of images
    #print(images[0].size()) # ---> torch.Size([1, 28, 28])     # images[0] is the image at index 0 in the given batch
    #print(images[0].squeeze(0).size())  # ---> torch.Size([28, 28])
    return images, labels


def show(x):
    reconstructed_image = model.reconstruction_from_posterior(x)
    plt.imshow(reconstructed_image.cpu().detach().numpy())
    plt.show()
    
def main():
    
    images, labels = get_data(train_dataloader)
    #images, labels = get_data(test_dataloader)
    
    index = 3  # ranges from 0 to batch_size
    x = images[index].squeeze(0)
    x = x.to(device)
    print("This image is of digit: ",labels[index].numpy())
    
    show(x)
    
main()
