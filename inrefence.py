import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torch import nn
import torch 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.model.model_architecture import UNET
import cv2
from torch.utils.data import Dataset
import torch

if __name__ == '__main__':
    unet = UNET(4,1)
    unet.load_state_dict(torch.load('unet.pth'))
    unet.eval()
    x = torch.randn(1, 4, 224, 224, requires_grad=True)
    torch_out = unet(x)
    pass