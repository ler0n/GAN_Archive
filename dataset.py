import os
import glob

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
    

class P2PDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self._preprocess(data_path)

    def _preprocess(self, data_path):
        self.x, self.y = [], []
        x_path = glob.glob(os.path.join(data_path, '*-0.png')) 
        y_path = glob.glob(os.path.join(data_path, '*-2.png')) 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        for x, y in zip(x_path, y_path):
            x_img = Image.open(x)
            self.x.append(transform(x_img))
            y_img = Image.open(y)
            self.y.append(transform(y_img))
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
