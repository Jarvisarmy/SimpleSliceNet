import torch

from collections import Counter
import torchvision
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from torch.utils.data import Dataset

import json
import os

import random
import gc
import sys
import copy
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def divide255(image, **kwargs):
    image = image / 255.0
    return image.astype('float32')


def get_transform(img_size, crop_size, train=True):
    if train:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.CenterCrop(crop_size, crop_size),
        ])
        return transform
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.CenterCrop(crop_size, crop_size),
        ])
        return transform


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for Fixmatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 img_paths,
                 targets=None,
                 transform=None,
                 train=True,
                 imagenet_norm=True,
                 *args, **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.img_paths = img_paths
        self.targets = targets
        self.transform = transform
        self.train = train
        self.totensor = A.Compose([
            A.Normalize() if imagenet_norm else A.Lambda(image=divide255),
            ToTensorV2()])

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target = self.targets[idx]

        # set augmented images
        img = default_loader(self.img_paths[idx])
        img = np.array(img)
        filename = os.path.basename(self.img_paths[idx])

        img_t = self.transform(image=img)['image']
        img_n = self.totensor(image=img_t)['image']
        #return idx, img_n, img_t, target, filename
        return {
            'image': img_n,
            'original': img_t,
            'label': target
        }

    def __len__(self):
        return len(self.img_paths)


class Br35H:
    """
    SSL_Dataset class gets dataset from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self,
                 name='brain-mri',
                 img_size=224,
                 crop_size=224,
                 train=True,
                 data_dir='../data/Br35H',
                 transform=None,
                 train_samples_limit=10000,
                 imagenet_norm=True
                 ):
        """
        Args
            alg: SSL algorithms
            name: name of dataset in torchvision.datasets (cifar10, cifar100, svhn, stl10)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        Returns:
            dataset: which outputs idx, img_n, img_t, target, filename
        """
        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.train_samples_limit = train_samples_limit
        self.imagenet_norm = imagenet_norm
        if transform is None:
            self.transform = get_transform(img_size, crop_size, train)
        else:
            self.transform = transform

    def get_data(self):
        """
        get_data returns data path and targets (labels)
        shape of img_paths: B
        shape of labels: B,
        """
        norm_path = os.path.join(self.data_dir, 'no')
        norm_files = [os.path.join(norm_path, file) for file in os.listdir(norm_path)]

        abnorm_path = os.path.join(self.data_dir, 'yes')
        abnorm_files = [os.path.join(abnorm_path, file) for file in os.listdir(abnorm_path)]

        if self.train:
            img_files = norm_files[:1000]
            targets = np.zeros(1000)
        else:
            img_files = norm_files[1000:]+abnorm_files
            targets = np.concatenate((np.zeros(500), np.ones(len(abnorm_files))))
        
        return img_files, targets

    def get_dset(self):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            strong_transform: list of strong transform (RandAugment in FixMatch)
            onehot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """

        img_paths, targets = self.get_data()
        dset = BasicDataset(img_paths, targets, transform=self.transform, imagenet_norm=self.imagenet_norm)
        return dset
