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

def get_image_plot(model, latent, width, height):
    model.eval()
    with torch.no_grad():
        gen_res = model(latent).detach().view(24, 1, width, height).cpu()
    
    img = torchvision.utils.make_grid(gen_res, nrow=6, normalize=True, padding=1)
    fig, ax = plt.subplots()
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')
    fig.tight_layout()
    return fig
