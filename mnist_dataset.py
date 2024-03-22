
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class mnist_data(Dataset):

    def __init__(self,transform=None):
        self.train_data = pd.read_csv('dataset/train.csv')
        self.target = self.train_data.pop('label')
        self.data = self.train_data
        self.transform = transform

        self.data = torch.Tensor(np.array(self.data)).reshape(-1,28,28).unsqueeze(1).float()
        self.target = torch.Tensor(np.array(self.target)).long()

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.target[index]

        # Apply transformations if specified
        if self.transform:
            sample = self.transform(sample)

        return sample, target
