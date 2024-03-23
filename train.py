import torch
import torch.nn as nn
from torch.types import Device
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from colorama import Fore

import torch.optim as optim
from tqdm import tqdm

#importing the custom mnist dataset
from mnist_dataset import mnist_data
from model import mnist_model
from util import plot_loss
from drive import upload

# Epoches = 100
# Learning_rate = 1e-3
# Batch_size = 4

def train(Epoches = 100,Learning_rate = 1e-4,Batch_size = 4):

    #configuring the device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(Fore.GREEN + "Device is CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(Fore.GREEN + "Device is MPS")
    else:
        device = torch.device('cpu')
        print("Device is running on cpu")

    #preparing the dataloader
    custom_dataloader = DataLoader(mnist_data(), batch_size=Batch_size, shuffle=True,pin_memory=True)

    #preparing the model
    model = mnist_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)

    # Train the model
    i = 0
    loss_list = []
    for epoch in tqdm(range(Epoches)):

        model.train()

        for data, target in custom_dataloader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

    torch.save(model.state_dict(), 'mnist_classifier.pt')

    try:
        upload('mnist_classifier.pt',"Mnist model")
    except: print(Fore.RED + "Could not upload the model to the drive")

    try:
        plot_loss(loss_list,save=True)
    except: print("could not upload the plot to drive")
