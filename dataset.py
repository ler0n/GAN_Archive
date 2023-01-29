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
	