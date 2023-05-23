import torch.nn as nn
import torch
import torchvision
from pytorch_resnet_cifar10.resnet import resnet20 as resnet20
from ct.src.cct import cct_7_3x1_32

class Mask(nn.Module):

    '''
    Class that inherits from torch.nn.Module, implementing the Fourier modulatory mask as a pre-processing layer

    Attribute M (torch.tensor): tensor that stores the entries of the mask
    '''

    def __init__(self, mask_size: tuple = (3, 32, 32)):
        super().__init__()
        assert len(mask_size)==3
        kernel = torch.ones((1, *mask_size))
        self.M = nn.Parameter(kernel)
        nn.init.ones_(self.M)

    def forward(self, x):
        x = torch.fft.fft2(x)
        x = self.M * x
        x = torch.fft.ifft2(x).real
        return x

class MaskedClf(nn.Module):

    '''
    Class that inherits from torch.nn.Module, implementing a end-to-end 'masked' classifier

    Attribute mask (Mask): pre-processing layer doing mask modulation
    Attribute clf (torch.nn.Module): pre-trained classifier
    '''

    def __init__(self, mask, clf):
        super().__init__()
        self.mask=mask
        self.clf=clf
    def forward(self, x):
        x=self.mask(x)
        x=self.clf(x)
        return x

def get_model(model_name):

    '''
    Utility function to obtain a classification model

    Param model_name (str): name of the model to load
    
    Return: Chosen model
    '''

    if model_name=='resnet20':
        model=resnet20()
    elif model_name=='cct':
        model = cct_7_3x1_32(pretrained=True)
    elif model_name=='resnet18':
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(512, 10)
        for n,p in model.named_parameters():
            if n!="fc.weight" and n!="fc.bias":
                p.requires_grad=False
    return model

