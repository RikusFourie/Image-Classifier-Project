import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
from workspace_utils import active_session
import numpy as np
from PIL import Image
from torch.autograd import Variable
import argparse
import os

from train_model import proc_data, initialise_model, train_model

parser = argparse.ArgumentParser()

parser.add_argument('data_directory', action='store',
                    default = 'flowers',
                    help='Set a file directory')

parser.add_argument('--save_dir', action='store',
                    default = '.',
                    dest='save_dir',
                    help='Set directory to save checkpoints')

parser.add_argument('--arch', action='store',
                    default = 'vgg11',
                    dest='arch',
                    help='Choose architecture for eg. vgg19')

parser.add_argument('--learning_rate', action='store',
                    default = 0.0001,
                    dest='lr',
                    help='Choose optimizer learning rate')

parser.add_argument('--hidden_units', action='store',
                    default = 4096,
                    dest='hidden_units',
                    help='Choose number of hidden unites')

parser.add_argument('--epochs', action='store',
                    default = 10,
                    dest='epochs',
                    help='Choose number of epochs during training')


parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for training, set a switch to true')

parse_results = parser.parse_args()

data_dir = parse_results.data_directory
save_dir = parse_results.save_dir
arch = parse_results.arch
lr = float(parse_results.lr)
hidden_units = int(parse_results.hidden_units)
epochs = int(parse_results.epochs)
gpu = parse_results.gpu

if gpu==True:
    device='cuda'
else:
    device='cpu'

image_datasets, image_dataloaders= proc_data(data_dir)
print('Datasets loaded...\n')

model, classifier = initialise_model(arch, hidden_units, device)
print('Model Initialised...\n')

print('Model Training...\n')

model, optimizer, criterion = train_model(model, lr, device, epochs, image_datasets, image_dataloaders)

# Saving Checkpoint
model.to('cpu')
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'arch': 'vgg19',
              'model': model,
              'state_dict': model.state_dict(),
              'classifier' : classifier,
              'epochs': epochs,
              'criterion': criterion,
              'optimizer': optimizer,
              'class_to_idx': model.class_to_idx}
 
if not os.path.exists(save_dir):
    os.mkdir(save_dir)  

torch.save(checkpoint, save_dir + '/checkpoint1.pth')
    

print(f'Checkpoint saved to {save_dir} folder.')