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
import random
from sklearn.metrics import roc_auc_score
# torch softmax
from torch.nn.functional import softmax

cudnn.benchmark = True
plt.ion()   # interactive mode

# current date and time
current_date_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# cannel the link
current_date_str = '2020-04-20-14-11-00'

if not os.path.exists('weights/{}'.format(current_date_str)):
    os.makedirs('weights/{}'.format(current_date_str))

if __name__ == '__main__':

    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([1024, 1024]),
        # transforms.RandomResizedCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.RandomRotation(degrees=15, expand=False, fill=None),
        # flip
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.37955950568704044, 0.23169069477156096, 0.10694904701382506], [0.2844342100854014, 0.17562707639208028, 0.0917047986797258])
    ]),
    'test': transforms.Compose([
        transforms.Resize([1024, 1024]),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.37955950568704044, 0.23169069477156096, 0.10694904701382506], [0.2844342100854014, 0.17562707639208028, 0.0917047986797258])
    ]),
    }


    data_dir = './dataset/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                shuffle=True, num_workers=2)
                for x in ['train','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','test']}
    class_names = image_datasets['train'].classes

    print(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    # dropout layer
    

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    # model_ft = model_ft.to(device)

    # load weights "wang_weights.pth"
    # model_ft.load_state_dict(torch.load('wang_weights.pt'))

    # dropout layer
    model_ft.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1),
    )

    model_ft = model_ft.to(device)


    model_ft= nn.DataParallel(model_ft)
    
    criterion = nn.BCEWithLogitsLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.9)
    
    max_auc = 0
    # train, calcuate auc sensitivity and specificity in each epoch
    for epoch in range(40, 1000):
        print('current epoch: {}'.format(epoch))
        exp_lr_scheduler.step()
        model_ft.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer_ft.zero_grad()
            outputs = model_ft(inputs)
            preds = softmax(outputs, 1)
            loss = criterion(outputs.squeeze().float(), labels.float())
            loss.backward()
            optimizer_ft.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        model_ft.eval()
        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_gts = []
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer_ft.zero_grad()
            with torch.no_grad():
                outputs = model_ft(inputs)
                all_preds.append(outputs.cpu().numpy())
                all_gts.append(labels.cpu().numpy())
                preds = softmax(outputs, 1)
                loss = criterion(outputs.squeeze().float(), labels.float())
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / dataset_sizes['test']
        epoch_acc = running_corrects.double() / dataset_sizes['test']
        auc = roc_auc_score(np.concatenate(all_gts), np.concatenate(all_preds))
        print('Val Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(epoch_loss, epoch_acc, auc))

        # create folder weights if not exist
        if not os.path.exists('weights'):
            os.makedirs('weights')

        if auc > max_auc:
            torch.save(model_ft.state_dict(), 'weights/{}/model_{}_{}_{}.pth'.format(current_date_str, epoch, epoch_acc, auc))
            max_auc = auc