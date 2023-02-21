import os
import glob
import torch
import random
import argparse
import torchvision
import numpy as np
import matplotlib.pyplot as plt 

from PIL import Image
from torchvision import transforms


def set_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_image_plot(model, title, latent, channel, width, height):
    model.eval()
    with torch.no_grad():
        gen_res = model(latent).detach().view(24, channel, width, height).cpu()
    model.train()
    img = torchvision.utils.make_grid(gen_res, nrow=6, normalize=True, padding=1)
    fig, ax = plt.subplots()
    ax.imshow(img.permute(1, 2, 0))
    ax.axis('off')
    fig.tight_layout()
    fig.suptitle(title)
    return fig

def get_image_plot2(model, title, data, truth):
    model.eval()
    with torch.no_grad():
        gen_res = model(data.unsqueeze(0)).squeeze(0).detach().cpu()
    model.train()
    imgs = [data.detach().cpu(), truth, gen_res]
    fig, axes = plt.subplots(1, 3)
    for ax, img in zip(axes, imgs):
        ax.imshow(img.permute(1, 2, 0))
        ax.axis('off')
    fig.tight_layout()
    fig.suptitle(title)
    return fig

def convert_path_to_tensor(data_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    img = Image.open(f'{data_path}-0.png')
    truth = Image.open(f'{data_path}-2.png')
    return transform(img), transform(truth)

def generate_gif_with_png(data_path, output_path, output_name, duration):
    image_paths = sorted(glob.glob(os.path.join(data_path, '*.png')), key=lambda x: int(x.split('_')[-2]))
    frames = [Image.open(path) for path in image_paths]
    frame_one = frames[0]
    frame_one.save(os.path.join(output_path, f'{output_name}.gif'), format='GIF', 
                   append_images=frames, save_all=True, duration=duration, loop=0)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # train option hyperparameter
    args.add_argument('--data_path', type=str, default='./images', help='path for output gif')
    args.add_argument('--output_path', type=str, default='./output', help='path for output gif')
    args.add_argument('--output_name', type=str, default='result', help='name for output gif')
    args.add_argument('--duration', type=int, default=500, help='duration for 1 image(millsec)')

    args = args.parse_args()
    generate_gif_with_png(args.data_path, args.output_path, args.output_name, args.duration)