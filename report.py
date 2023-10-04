from __future__ import print_function, division

import time
import os
import copy
import random
import glob
# warning
import warnings
# hide warnings
warnings.filterwarnings('ignore')

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn.functional import softmax, sigmoid
from utils import *

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, imagepaths, transform):
        self.transform = transform
        self.imagepaths = imagepaths
    
    def __len__(self):
        return len(self.imagepaths)
    
    def __getitem__(self, idx):
        img = Image.open(self.imagepaths[idx])
        img = self.transform(img)
        label = self.imagepaths[idx].split('/')[-2]
        if label == '1':
            label = 1
        else:
            label = 0
        return img, label, self.imagepaths[idx]

cudnn.benchmark = True
plt.ion()   # interactive mode

if __name__ == '__main__':
    while True:
        target_dir = 'Photos'
        # read from config.txt to get the target image path
        try:
            with open('config.txt', 'r') as f:
                target_dir = f.read().split('\n')[0]
        except:
            pass
        print("target_dir: ", target_dir)

        to_save_dir = 'Reports'
        if not os.path.exists(to_save_dir):
            os.makedirs(to_save_dir)

        # detect which reports have not been generated in the to_save_dir
        target_image_paths = []
        for target_image_path in glob.glob(os.path.join(target_dir, '*.jpg')):
            if not os.path.exists(os.path.join(to_save_dir, os.path.basename(target_image_path))):
                target_image_paths.append(target_image_path)

        for target_image_path in target_image_paths:
            print(target_image_path)
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

            all_imagepaths = {'train': [target_image_path],
                                'test': [target_image_path]}

            data_dir = './dataset/'
            image_datasets = {x: CustomDataset(all_imagepaths[x], data_transforms[x]) for x in ['train','test']}

            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                        shuffle=True, num_workers=2) for x in ['train','test']}

            dataset_sizes = {x: len(image_datasets[x]) for x in ['train','test']}
            class_names = ['0', '1']

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model_ft_ = models.resnet34(pretrained=True)
            num_ftrs = model_ft_.fc.in_features

            # dropout layer
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
            # model_ft.fc = nn.Linear(num_ftrs, 2)
            # model_ft = model_ft.to(device)

            # load weights "wang_weights.pth"
            # model_ft.load_state_dict(torch.load('wang_weights.pt'))

            # dropout layer
            model_ft_.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 1),
            )

            model_ft_ = model_ft_.to(device)
            model_ft_ = nn.DataParallel(model_ft_)

            # load weights
            model_ft_.load_state_dict(torch.load('/home/philip/Documents/Philip_4T/Glaucoma_fundus/x-resnet/FundusGlaucomaEidon/weights/2020-04-20-14-11-00/model_162_8.60377358490566_0.8677536231884058.pth'))
            
            model_ft = model_ft_.module
            cam = GradCAM(model=model_ft, target_layers=[model_ft.layer4[-1]])

            model_ft_.eval()
            for inputs, labels, imagepths in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = model_ft_(inputs)
                    outputs = sigmoid(outputs)
                    epoch_preds = outputs.clone().detach().cpu().numpy().tolist()[0]


                targets = [BinaryClassifierOutputTarget(1)]
                grayscale_cams = cam(input_tensor=inputs, targets=targets)
                # In this example grayscale_cam has only one image in the batch:
                for grayscale_cam, imagepth, epoch_pred in zip(grayscale_cams, imagepths, epoch_preds):
                    rgb_img = Image.open(imagepth)
                    rgb_img = rgb_img.resize((1024, 1024))
                    raw_rgb_img = rgb_img.copy()
                    rgb_img = np.array(rgb_img, dtype=np.float32) / 255
                    rgb_img = rgb_img[:,:,:3]

                    # rgb
                    if epoch_pred > 0.5:
                        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    else:
                        visualization = rgb_img

                    raw_rgb_img = np.array(np.array(raw_rgb_img, dtype=np.float32) / np.max(raw_rgb_img)*255, dtype=np.uint8)
                    visualization = np.array(np.array(visualization, dtype=np.float32) / np.max(visualization)*255, dtype=np.uint8)
                    create_report(os.path.basename(imagepth), epoch_pred, raw_rgb_img, visualization, to_save_dir)


