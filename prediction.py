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

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    classifier=checkpoint['classifier']
    model.classifier = classifier
    criterion=checkpoint['criterion']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer=checkpoint['optimizer']
    class_to_idx=checkpoint['class_to_idx']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model,optimizer,criterion,class_to_idx,device


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    II = Image.open(image)
    II.load()

    if II.size[0] > II.size[1]:
        II.thumbnail((100000, 256))
    else:
        II.thumbnail((256, 100000))

    left = (II.size[0] - 224) / 2
    lower = (II.size[1] - 224) / 2
    right = (II.size[0] + 224) / 2
    upper = (II.size[1] + 224) / 2
    cropped_img = II.crop((left, lower, right,
                           upper))

    np_img = np.array(cropped_img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    Std_image = (np_img - mean) / std
    trans_img = Std_image.transpose((2, 0, 1))
    return trans_img


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, device, topk, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()

    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = image.type(torch.FloatTensor)
        image = image.to(device)

        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_c = torch.topk(ps, topk)

        tp = []
        for p in top_p[0]:
            tp.append(float(p))

        tc = []
        for c in top_c[0]:
            tc.append(float(c))

        cti = dict(model.class_to_idx.items())

        ind = []
        for i in tc:
            ind.append(list(cti.keys())[list(cti.values()).index(i)])
        flower_names = []
        for i in ind:
            flower_names.append(cat_to_name[i])
    return tp, ind, flower_names