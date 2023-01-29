import torch
import random
import numpy as np

import matplotlib.pyplot as plt 


def set_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_image_plot(model, device, noise_dim):
	model.eval()
	with torch.no_grad():
		noise = torch.FloatTensor(size=(24, noise_dim), device=device).normal_(0, 1)
		gen_res = model(noise).view(24, 28, 28).cpu().numpy()

	fig, axes = plt.subplots(4, 6, figsize=(8, 4))
	axes = axes.flatten()
	for i in range(24):
		axes[i].imshow(gen_res[i], interpolation='nearest', cmap='gray')
		axes[i].axis('off')
	return fig
