import torch
import random
import torchvision
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
        noise = torch.FloatTensor(size=(24, noise_dim)).normal_(0, 1).to(device)
        gen_res = model(noise).detach().view(24, 28, 28).unsqueeze(1).cpu()
    
    img = torchvision.utils.make_grid(gen_res, nrow=6, normalize=True, padding=1)
    fig, ax = plt.subplots()
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')
    fig.tight_layout()
    return fig
