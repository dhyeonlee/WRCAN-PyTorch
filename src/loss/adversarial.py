import utility
from types import SimpleNamespace

from model import common
from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.disc_arch = args.gan_arch.lower()

        if self.disc_arch.find('rp') >= 0:
            self.dis = discriminator.Discriminator_RP(args)
        elif self.disc_arch.find('patch') >= 0:
            self.dis = discriminator.Discriminator_patch(args)
        elif self.disc_arch.find('kernel') >= 0:
            self.dis = discriminator.Discriminator_kernel(args)
        else: # Vinalla discriminator
            self.dis = discriminator.Discriminator(args)
        
        # self.dis = discriminator.Discriminator(args) if gan_type.find('RP') < 0 else discriminator.Discriminator_RP(args)
        if gan_type == 'WGAN_GP':
            # see https://arxiv.org/pdf/1704.00028.pdf pp.4
            optim_dict = {
                'optimizer': 'ADAM',
                'betas': (0, 0.9),
                'epsilon': 1e-8,
                'lr': 1e-5,
                'weight_decay': args.weight_decay,
                'decay': args.decay,
                'gamma': args.gamma
            }
            optim_args = SimpleNamespace(**optim_dict)
        else:
            optim_args = args

        self.optimizer = utility.make_optimizer(optim_args, self.dis)


    def forward(self, fake, real):
        # updating discriminator...
        self.loss = 0
        fake_detach = fake.detach()     # do not backpropagate through G
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            # d: B x 1 tensor
            d_fake = self.dis(fake_detach)
            d_real = self.dis(real)
            retain_graph = False
            if self.gan_type == 'GAN':
                loss_d = self.bce(d_real, d_fake)
            elif self.gan_type == 'LSGAN':
                loss_d = self.mse(d_real, d_fake)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.dis(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
            # from ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
            elif self.gan_type == 'RGAN':
                better_real = d_real - d_fake.mean(dim=0, keepdim=True)
                better_fake = d_fake - d_real.mean(dim=0, keepdim=True)
                loss_d = self.bce(better_real, better_fake)
                retain_graph = True

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward(retain_graph=retain_graph)
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.dis.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        # updating generator...
        d_fake_bp = self.dis(fake)      # for backpropagation, use fake as it is
        if self.gan_type == 'GAN':
            label_real = torch.ones_like(d_fake_bp)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_bp, label_real)
        elif self.gan_type == 'LSGAN':
            label_real = torch.ones_like(d_fake_bp)
            loss_g = F.mse_loss(d_fake_bp, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_bp.mean()
        elif self.gan_type == 'RGAN':
            better_real = d_real - d_fake_bp.mean(dim=0, keepdim=True)
            better_fake = d_fake_bp - d_real.mean(dim=0, keepdim=True)
            loss_g = self.bce(better_fake, better_real)

        # Generator loss
        return loss_g
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()
        
        return dict(**state_discriminator, **state_optimizer)

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if 'features' in name:
                name = name[name.find('features'):]
            elif 'classifier' in name:
                name=name[name.find('classifier'):]
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('classifier') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('classifier') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

    
    def bce(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        bce_real = F.binary_cross_entropy_with_logits(real, label_real)
        bce_fake = F.binary_cross_entropy_with_logits(fake, label_fake)
        bce_loss = bce_real + bce_fake
        return bce_loss

    def mse(self, real, fake):
        label_real = torch.ones_like(real)
        label_fake = torch.zeros_like(fake)
        mse_real = F.mse_loss(real, label_real)
        mse_fake = F.mse_loss(fake, label_fake)
        mse_loss = 0.5*(mse_real + mse_fake)
        return mse_loss
               
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
