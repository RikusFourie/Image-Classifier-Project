import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import json
from workspace_utils import active_session
import numpy as np
from PIL import Image
from torch.autograd import Variable
import argparse
from prediction import process_image, imshow, predict, load_checkpoint

parser = argparse.ArgumentParser()

parser.add_argument('image_path', action='store',
                    default = 'flowers/test/102/image_08042.jpg',
                    help='File path to image')

parser.add_argument('checkpoint', action='store',
                    default = 'checkpoint.pth',
                    help='Directory of saved checkpoint')

parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Use GPU for training, set a switch to true')


parser.add_argument('--top_k', action='store', type=int,
                    default = 5,
                    dest='top_k',
                    help='Return class/classes with the highest possibility')

parser.add_argument('--category_names', action='store',
                    default = 'cat_to_name.json',
                    dest='cat_names',
                    help='File name of the mapping of flower categories to real names, e.g., "cat_to_name.json"')

parse_results = parser.parse_args()

image_path = parse_results.image_path
checkpoint = parse_results.checkpoint
top_k = parse_results.top_k
category_names = parse_results.cat_names
gpu = parse_results.gpu

if gpu==True:
    device='cuda'
else:
    device='cpu'
        
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
nn_filename = checkpoint
actual_class = image_path.split('/')[2]
titl=cat_to_name[actual_class]

checkpoint = torch.load(nn_filename)
model = checkpoint["model"]
model.load_state_dict(checkpoint['state_dict'])
print('Model Loaded...\n')

probs, classes,flower_names = predict(image_path, model, device, top_k, cat_to_name)

print('Actual flower: {}'.format(titl))
print('Predicted flower: {}\n'.format(flower_names[0]))

for i in range(top_k):
    print('{}.Predicted flower: {} ---- Probability: {}'.format(i+1,flower_names[i],round(probs[i],2)*100))