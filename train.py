import os
import torch
import argparse
import configparser

from model import *
from loss import P2PGeneratorLoss, P2PDiscriminatorLoss
from dataset import BaseDataset, DCGANDataset, P2PDataset
from logger import BaseLogger, WandbLogger, NeptuneLogger
from util import set_seed, get_image_plot, get_image_plot2, convert_path_to_tensor

from tqdm import tqdm
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import DataLoader


def get_model_and_dataloader(args):
    dataloader, generator, discriminator = None, None, None

    if args.model_type.lower() == 'gan':
        dataset = BaseDataset()
        image_dim = dataset[0].shape[0]
        args.gen_layer_dim = [args.latent_dim] + args.gen_layer_dim + [image_dim]
        args.dis_layer_dim = [image_dim] + args.dis_layer_dim + [1]
        generator = BaseGenerator(args.gen_layer_dim)
        discriminator = BaseDiscriminator(args.dis_layer_dim)
    if args.model_type.lower() == 'dcgan':
        dataset = DCGANDataset(args.data_path)
        generator = DCGenerator(args.latent_dim, args.channel_num, args.channel_list)
        discriminator = DCDiscriminator(args.channel_num, args.channel_list, args.dropout_rate)
    if args.model_type.lower() == 'pix2pix':
        dataset = P2PDataset(args.data_path)
        generator = P2PGenerator(args.channel_num, args.channel_list, args.dropout_rate)
        discriminator = P2PDiscriminator(args.channel_num, args.channel_list)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    return generator, discriminator, dataloader

def i2i_gan_train(args, logger):
    device = f'cuda:{args.device_num}' if torch.cuda.is_available() else 'cpu'
    test_img, test_true = convert_path_to_tensor(args.test_img_path)
    test_img = test_img.to(device)

    generator, discriminator, dataloader = get_model_and_dataloader(args)
    generator.to(device)
    discriminator.to(device)
    
    gen_loss = P2PGeneratorLoss(args.lmd).to(device)
    dis_loss = P2PDiscriminatorLoss().to(device)
    gen_optimizer = Adam(generator.parameters(), lr=args.lr_rate, betas=(args.beta, 0.999))
    dis_optimizer = Adam(discriminator.parameters(), lr=args.lr_rate, betas=(args.beta, 0.999))

    for epoch in range(1, args.epoch + 1):
        tbar = tqdm(dataloader)
        tbar.set_description(f'Epoch {epoch}')
        d_loss, g_loss = 0, 0
        for x, y in tbar:
            x = x.to(device)
            y = y.to(device)
            for _ in range(args.dis_iter):
                discriminator.zero_grad()
                
                y_in = torch.cat([y, y], axis=1)
                pos_res = discriminator(y_in)

                with torch.no_grad():
                    gen_res = generator(x)
                
                gen_res = torch.cat([gen_res, y], axis=1)
                neg_res = discriminator(gen_res)
                disc_loss = dis_loss(pos_res, neg_res)
                disc_loss.backward()
                d_loss += disc_loss.detach().cpu().item()
                dis_optimizer.step()

            generator.zero_grad()
    
            gen_res = generator(x)
            gen_res = torch.cat([gen_res, y], axis=1)
            dis_res = discriminator(gen_res)
            gene_loss = gen_loss(dis_res, gen_res, y)
            gene_loss.backward()
            g_loss += gene_loss.detach().cpu().item()
            gen_optimizer.step()
        
        g_loss /= len(dataloader)
        d_loss /= len(dataloader)
        loss_dict = {
            'g_loss': g_loss,	
            'd_loss': d_loss
        }
        tbar.set_postfix(loss_dict)
        logger.write_log(loss_dict)

        if epoch != args.epoch and epoch % args.save_interval != 0: continue
        fig = get_image_plot2(generator, test_img, test_true)
        fig.suptitle(f'Epoch {epoch} result')
        logger.write_figure(epoch, fig)
        torch.save(generator.state_dict(), os.path.join(args.save_path, 'p2_generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(args.save_path, 'p2_discriminator.pth'))

def train(args, logger):
    device = f'cuda:{args.device_num}' if torch.cuda.is_available() else 'cpu'
    latent_test = torch.FloatTensor(size=(24, args.latent_dim)).normal_(0, 1).to(device)

    generator, discriminator, dataloader = get_model_and_dataloader(args)
    generator.to(device)
    discriminator.to(device)
    
    loss = BCELoss()
    gen_optimizer = Adam(generator.parameters(), lr=args.lr_rate, betas=(args.beta, 0.999))
    dis_optimizer = Adam(discriminator.parameters(), lr=args.lr_rate, betas=(args.beta, 0.999))

    for epoch in range(1, args.epoch + 1):
        tbar = tqdm(dataloader)
        tbar.set_description(f'Epoch {epoch}')
        d_loss, g_loss = 0, 0
        for x in tbar:
            x = x.to(device)
            for _ in range(args.dis_iter):
                discriminator.zero_grad()

                pos_res = discriminator(x)
                pos_label = torch.ones(x.shape[0], device=device)
                pos_loss= loss(pos_res, pos_label)
                pos_loss.backward()

                latent = torch.FloatTensor(size=(x.shape[0], args.latent_dim)).normal_(0, 1).to(device)
                with torch.no_grad():
                    gen_res = generator(latent)
                neg_res = discriminator(gen_res)
                neg_label = torch.zeros(x.shape[0], device=device)
                neg_loss = loss(neg_res, neg_label)
                neg_loss.backward()
                dis_loss = neg_loss + pos_loss
                d_loss += dis_loss.detach().cpu().item()
                dis_optimizer.step()

            generator.zero_grad()
    
            latent = torch.FloatTensor(size=(x.shape[0], args.latent_dim)).normal_(0, 1).to(device)
            dis_label = torch.ones(x.shape[0]).to(device)
            gen_res = generator(latent)
            dis_res = discriminator(gen_res)
            gen_loss = loss(dis_res, dis_label)

            gen_loss.backward()
            g_loss += gen_loss.detach().cpu().item()
            gen_optimizer.step()
        
        g_loss /= len(dataloader)
        d_loss /= len(dataloader)
        loss_dict = {
            'g_loss': g_loss,	
            'd_loss': d_loss
        }
        tbar.set_postfix(loss_dict)
        logger.write_log(loss_dict)

        if epoch != args.epoch and epoch % args.save_interval != 0: continue
        fig = get_image_plot(generator, latent_test, args.channel_num, args.width, args.height)
        fig.suptitle(f'Epoch {epoch} result')
        logger.write_figure(epoch, fig)
        torch.save(generator.state_dict(), os.path.join(args.save_path, 'generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(args.save_path, 'discriminator.pth'))
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # train option hyperparameter
    args.add_argument('--epoch', type=int, default=50, help='epoch')
    args.add_argument('--batch_size', type=int, default=128, help='batch size')
    args.add_argument('--lr_rate', type=float, default=2e-4, help='learning rate')
    args.add_argument('--beta', type=float, default=5e-1, help='beta value')
    args.add_argument('--width', type=int, default=64, help='width value')
    args.add_argument('--height', type=int, default=64, help='height value')
    args.add_argument('--dis_iter', type=int, default=1, 
                      help='discriminator train iteration per epoch')

    # common model hyperparameter
    args.add_argument('--latent_dim', type=int, default=100, help='noise dimension')
    args.add_argument('--dropout_rate', type=float, default=0.3, help='ratio of dropout')
    
    # for vanilla gan
    args.add_argument('--gen_layer_dim', type=list, default=[256, 512, 1024], 
                      help='generator layer dimensions')
    args.add_argument('--dis_layer_dim', type=list, default=[1024, 512, 256], 
                      help='discriminator layer dimensions')
    
    # for dcgan, pix2pix
    args.add_argument('--channel_num', type=int, default=1, 
                      help='number of output data channel')
    args.add_argument('--channel_list', type=list, default=[16, 64, 128, 256, 512], 
                      help='model layer channels')
    args.add_argument('--lmd', type=int, default=100,
                      help='value of lambda for calculate generator loss')
    
    # log, save interval, path, model type option
    args.add_argument('--model_type', type=str, default='pix2pix', 
                      help='model type to train(gan, dcgan, pix2pix)')
    args.add_argument('--log_type', type=int, default=2, 
                      help='0 = not log, 1 = neptune, 2 = wandb')
    args.add_argument('--save_interval', type=int, default=15, 
                      help='interval for saving model state')
    args.add_argument('--data_path', type=str, default='./data/letters', 
                      help='path for train data')
    args.add_argument('--save_path', type=str, default='./saved', 
                      help='path for save model parameters')
    args.add_argument('--test_img_path', type=str, default='./data/letters/한',
                      help='path for test image')

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
    if args.model_type.lower() != 'pix2pix': train(args, logger)
    else: i2i_gan_train(args, logger)

    logger.finish()
