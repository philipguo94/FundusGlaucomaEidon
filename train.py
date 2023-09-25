from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from process_utils import *

cudnn.benchmark = True
plt.ion()   # interactive mode


if __name__ == '__main__':

    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([448, 448]),
        # transforms.RandomResizedCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.RandomRotation(degrees=15, expand=False, fill=None),
        transforms.ToTensor(),
        transforms.Normalize([0.37955950568704044, 0.23169069477156096, 0.10694904701382506], [0.2844342100854014, 0.17562707639208028, 0.0917047986797258])
    ]),
    'val': transforms.Compose([
        transforms.Resize([448, 448]),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.37955950568704044, 0.23169069477156096, 0.10694904701382506], [0.2844342100854014, 0.17562707639208028, 0.0917047986797258])
    ]),
    }


    data_dir = '../dataset/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train','val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=96,
                                                shuffle=True, num_workers=4)
                for x in ['train','val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)
    model_ft= nn.DataParallel(model_ft)
    
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.8)

    model_ft = train_model(model_ft, dataloaders, device, criterion, optimizer_ft, exp_lr_scheduler,dataset_sizes, num_epochs=200)
