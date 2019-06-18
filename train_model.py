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

def proc_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train':datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid':datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test':datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    image_dataloaders = {
        'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test':torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    }
    return image_datasets, image_dataloaders
    
def initialise_model(arch, hidden_units, device):
    # Load pretrained model
    model = getattr(models, arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('drop', nn.Dropout(p=0.5)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    model.to(device)
    
    return model, classifier 


def train_model(model, lr, device, epochs, image_datasets, image_dataloaders):
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    start_time = time.time()
    with active_session():
        for epoch in range(epochs):
            for step in ['train','valid']:
                running_loss = 0
                accuracy = 0
                batch_counter=0
                with torch.set_grad_enabled(step == 'train'):
                    if step == 'train':
                        model.train()
                    else:
                        model.eval()
                    for inputs, labels in image_dataloaders[step]:
                        batch_counter+=1
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()
                        if step=='train':
                            outputs=model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                        else:
                            outputs=model(inputs)
                            loss = criterion(outputs, labels)                   
                        running_loss += loss.item()
                        ps = torch.exp(outputs).data
                        top_p, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    if step == 'train':
                        print('\nEpoch:{}/{}\n--------------------------------'.format(epoch+1,epochs))
                        print('Training Loss: {}'.format(round(running_loss/batch_counter,4)))
                    else:
                        print('Validation Loss: {}'.format(round(running_loss/batch_counter,4)))
                        print('Accuracy: {}%'.format(round(accuracy*100/batch_counter,2)))

    end_time = time.time() - start_time 
    print("\nRun time was {:.0f}m and {:.0f}s".format(end_time//60, end_time % 60))
    return model, optimizer, criterion
    
    
    
