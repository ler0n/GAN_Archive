import torch
import argparse
import configparser

from dataset import BaseDataset
from util import set_seed, get_image_plot
from model import BaseGenerator, BaseDiscriminator
from logger import BaseLogger, WandbLogger, NeptuneLogger

from tqdm import tqdm
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader


def train(args, logger):
    device = f'cuda:{args.device_num}' if torch.cuda.is_available() else 'cpu'

    dataset = BaseDataset()
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    image_dim = dataset[0].shape[0]
    args.gen_layer_dim = [args.noise_dim] + args.gen_layer_dim + [image_dim]
    args.dis_layer_dim = [image_dim] + args.dis_layer_dim + [1]

    generator = BaseGenerator(args.gen_layer_dim).to(device)
    discriminator = BaseDiscriminator(args.dis_layer_dim, args.dropout_rate).to(device)
    
    loss = BCELoss()
    gen_optimizer = Adam(generator.parameters(), lr=args.lr_rate, betas=(args.beta, 0.999))
    dis_optimizer = Adam(discriminator.parameters(), lr=args.lr_rate, betas=(args.beta, 0.999))

    for epoch in range(1, args.epoch + 1):
        tbar = tqdm(dataloader)
        tbar.set_description(f'Epoch {epoch}')
        for x in tbar:
            x = x.to(device)
            for _ in range(args.dis_iter):
                gen_optimizer.zero_grad()
                dis_optimizer.zero_grad()

                generator.eval()
                discriminator.train()

                noise = torch.FloatTensor(size=(x.shape[0], args.noise_dim)).uniform_(-1, 1).to(device)
                gen_res = generator(noise)

                x_dis = torch.cat((x, gen_res), dim=0)
                label = torch.zeros(x.shape[0] * 2, device=device)
                label[:x.shape[0]] = 0.9

                dis_res = discriminator(x_dis)
                d_loss = loss(dis_res, label)

                d_loss.backward()
                dis_optimizer.step()
            
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            generator.train()
            discriminator.eval()
    
            noise = torch.FloatTensor(size=(x.shape[0], args.noise_dim)).uniform_(-1, 1).to(device)
            gen_label = torch.ones(x.shape[0]).to(device)
            gen_res = generator(noise)
            gen_res = discriminator(gen_res)
            g_loss = loss(gen_res, gen_label)

            g_loss.backward()
            gen_optimizer.step()
        
        gen_loss = g_loss.detach().cpu().item()
        dis_loss = d_loss.detach().cpu().item()
        loss_dict = {
            'g_loss': gen_loss,	
            'd_loss': dis_loss
        }
        tbar.set_postfix(loss_dict)
        logger.write_log(loss_dict)

        if epoch % 5 != 0: continue
        fig = get_image_plot(generator, device, args.noise_dim)
        fig.suptitle(f'Epoch {epoch} result')
        logger.write_figure(epoch, fig)
        
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # train option hyperparameter
    args.add_argument('--epoch', type=int, default=50, help='epoch')
    args.add_argument('--batch_size', type=int, default=128, help='batch size')
    args.add_argument('--lr_rate', type=float, default=2e-4, help='learning rate')
    args.add_argument('--beta', type=float, default=5e-1, help='beta value')
    args.add_argument('--dis_iter', type=int, default=1, 
                      help='discriminator train iteration per epoch')

    # model hyperparameter
    args.add_argument('--noise_dim', type=int, default=10, help='noise dimension')
    args.add_argument('--gen_layer_dim', type=list, default=[256, 512, 1024], 
                      help='generator layer dimensions')
    args.add_argument('--dis_layer_dim', type=list, default=[1024, 512, 256], 
                      help='discriminator layer dimensions')
    args.add_argument('--dropout_rate', type=float, default=0.3, help='ratio of dropout')
    
    # log option
    args.add_argument('--log_type', type=int, default=2, 
                      help='0 = not log, 1 = neptune, 2 = wandb')

    # gpu device option
    args.add_argument('--device_num', type=int, default=0, 
                      help='number of gpu device to use')

    args = args.parse_args()

    # load secrets
    config = configparser.ConfigParser()
    config.read('secret.ini')
    
    if args.log_type == 1: logger = NeptuneLogger(config, args)
    elif args.log_type == 2: logger = WandbLogger(config, args)
    else: logger = BaseLogger()
        
    set_seed(42)
    train(args, logger)

    logger.finish()
