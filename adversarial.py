from typing import Tuple, List, Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, CrossEntropyLoss
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

def ufgsm_attack(model: Union[Module, Sequential],
                images: Tensor, 
                labels: Tensor,
                criterion=CrossEntropyLoss(),
                epsilon:float = 0.1):
    ''' The untargeted fast gradient sign method from "Explaining and Harnessing Adversarial Examples" (<https://arxiv.org/pdf/1412.6572.pdf>). 
    The code is adapted from <https://pytorch.org/tutorials/beginner/fgsm_tutorial.html> 

    perturbed_image = image + epsilon ∗ sign(data_grad)

    Args:
        epsilon: the step size of FGSM. Intuitively we would expect the larger the epsilon, the more noticeable the perturbations but the more effective the attack in terms of degrading model accuracy.
    '''
    norm_min, norm_max = images.min(), images.max()
    images.requires_grad = True 

    logits = model(images)
    loss = criterion(logits, labels)
    model.zero_grad()
    loss.backward()

    # Collect the element-wise sign of the data gradient
    sign_data_grad = images.grad.data.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = images + epsilon*sign_data_grad
    # Adding clipping to pretain perturbed image in original scale
    perturbed_image = torch.clamp(perturbed_image, norm_min, norm_max)
    # Return the perturbed image
    return perturbed_image

