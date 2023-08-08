from torchvision.transforms import ToTensor
import torch
import os
from torch.utils.data import DataLoader,TensorDataset
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from optparse import OptionParser
import torch.nn.functional as F
import csv
import math
from PIL import Image
import torchvision
import numpy as np
from Net.DICNet_corr import DICNet
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = DICNet().to(device)
weightpath = 'weight_large_dis/DICNet-corr-unsuper.pth'
checkpoint = torch.load(weightpath)
net.load_state_dict(checkpoint['state_dict'])

net.eval()
with torch.no_grad():
    ref_image = Image.open('ref.png').convert('L')
    tar_image = Image.open('dis.png').convert('L')
    ref_image = torchvision.transforms.ToTensor()(ref_image)
    tar_image = torchvision.transforms.ToTensor()(tar_image)
    image = torch.cat((ref_image, tar_image), dim=0).unsqueeze(dim=0).to(device)
    image = torchvision.transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])(image)

    dis = net(image).cpu()

    plt.subplot(121)
    plt.imshow(dis[0][0].numpy(), cmap='jet')
    plt.subplot(122)
    plt.imshow(dis[0][1].numpy(), cmap='jet')
    plt.savefig("test.jpg")

