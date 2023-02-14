import os
import glob
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset).__init__()
        self.data = datasets.MNIST(
            root='data/',
            train=True,
            download=True
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, _ = self.data[idx]
        return self.transform(x).view(-1)


class DCGANDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self._preprocess(data_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
    
    def _preprocess(self, data_path):
        self.data = []
        data_path = glob.glob(os.path.join(data_path, '*.png')) 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        for path in data_path:
            img = Image.open(path)
            self.data.append(transform(img))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]