from typing import Tuple, List, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Sequential, CrossEntropyLoss
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import (Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, CenterCrop, Resize,
                                    RandomResizedCrop, RandomAffine, ColorJitter, RandomRotation)
from torchvision.datasets import MNIST, KMNIST, CIFAR10, SVHN, ImageFolder

import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets

### toy data generator 

def toy_classification_gen(random_seed=2222, train_size=500, test_size=50, centers=5, cluster_std=1.2, train_range=(-10, 10), test_range=(-15,15), random_state=33): 
    ### note sklearn uses numpy.random throughout, therefore use np.random.seed() here
    np.random.seed(random_seed)
    X, Y = datasets.make_blobs(n_samples=train_size, centers=centers, cluster_std=cluster_std, center_box=train_range, random_state=random_state)
    test_rng = np.linspace(*test_range, test_size)
    X1_test, X2_test = np.meshgrid(test_rng, test_rng)
    return X, Y, X1_test, X2_test

def toy_regression_gen(n_samples=100, X_range=(-4, 4), Y_noise_range=(0,9), sorted=False):
    # y = x^3
    data = torch.Tensor()
    X_Sampler = torch.distributions.uniform.Uniform(X_range[0], X_range[1])
    Y_Sampler = torch.distributions.normal.Normal(Y_noise_range[0],Y_noise_range[1])
    for i in range(n_samples):
        x = X_Sampler.sample() 
        y = x*x*x+Y_Sampler.sample()
        data = torch.cat([data, Variable(torch.tensor([x,y],dtype=torch.float)).reshape(1,-1)],dim=0) 
    return data[data[:,0].sort()[1]] if sorted == True else data

### load formal data
def mnist(root: str= "~/.torch/datasets", 
          batch_size: int=32, 
          split: str="train") -> DataLoader:
    ### train_split = True -> training dataset, otherwise test dataset
    if 'train' in split:
        train_set = MNIST(root, train=True, transform=ToTensor(), download=True)
        # set pin_memory to true to speed up GPU training
        data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    elif 'test' in split:
        test_set = MNIST(root, train=False, transform=ToTensor(), download=True)
        data_loader = DataLoader(test_set, batch_size=256)
    else: 
        raise TypeError
    return data_loader

def kmnist(root: str= "~/.torch/datasets", 
          batch_size: int=32, 
          split: str="train") -> DataLoader:
    ### Kuzushiji-MNIST Dataset <https://pytorch.org/vision/main/generated/torchvision.datasets.KMNIST.html>
    if 'train' in split:
        train_set = KMNIST(root, train=True, transform=ToTensor(), download=True)
        data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    elif 'test' in split:
        test_set = KMNIST(root, train=False, transform=ToTensor(), download=True)
        data_loader = DataLoader(test_set, batch_size=256)
    else: 
        raise TypeError
    return data_loader


def cifar10(root: str= "~/.torch/datasets", 
          batch_size: int=128, 
          split: str="train") -> DataLoader:
    normalize = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    transform = Compose([ToTensor(), normalize])

    if 'train' in split:
        train_set = CIFAR10(root, train=True, transform=transform, download=True)
        data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    elif 'test' in split:
        test_set = CIFAR10(root, train=False, transform=transform, download=True)
        data_loader = DataLoader(test_set, batch_size=256)
    else: 
        raise TypeError
    return data_loader

def svhn(root: str= "~/.torch/datasets", 
          batch_size: int=128, 
          split: str="train") -> DataLoader:
    normalize = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    transform = Compose([ToTensor(), normalize])
    
    if 'train' in split:
        train_set = SVHN(root, train=True, transform=transform, download=True)
        data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    elif 'test' in split:
        test_set = SVHN(root, train=False, transform=transform, download=True)
        data_loader = DataLoader(test_set, batch_size=256)
    else: 
        raise TypeError
    return data_loader