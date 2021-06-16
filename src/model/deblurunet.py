# -*- coding: utf-8 -*-
# @Author: Donghyeon Lee (donghyeon1223@gmail.com)
# @Date:   2021-01-16 11:01:15
# @Last Modified by:   Donghyeon Lee (donghyeon1223@gmail.com)
# @Last Modified time: 2021-03-31 22:40:05
from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return DeblurUNet(args)

class EncBlock(nn.Module):
    def __init__(self, conv, base_block, n_feats_in, n_feats_out, bn=False, act=nn.ReLU(True)):
        super(EncBlock, self).__init__()
        ksize = 3 # kernel size
        m_layers = [
            conv(n_feats_in, n_feats_out, ksize), # Expansion layer.
            act,
            base_block(conv, n_feats_out, n_feats_out, ksize, bias=not bn, bn=bn, act=act), # Feature transform
        ]
        m_pool = [
            nn.Conv2d(n_feats_out, n_feats_out, ksize, stride=2, padding=ksize//2, bias=True)
        ]
        self.layers = nn.Sequential(*m_layers)
        self.pool = nn.Sequential(*m_pool)

    def forward(self, x):
        feats = self.layers(x) # in: [B,n_feats_in, H, W], out: [B, n_feats_out, H, W]
        pool = self.pool(feats) # in :[B, n_feats_out, H, W], out: [B, n_feats_out, H/2, W/2]

        return pool, feats

class DecBlock(nn.Module):
    def __init__(self, conv, base_block, n_feats_in, n_feats_sc, n_feats_out, bn=False, act=nn.ReLU(True)):
        super(DecBlock, self).__init__()
        ksize = 3
        m_upsample = [common.Upsampler(conv, 2, n_feats_in, n_feats_out)]

        m_layers = [
            conv(n_feats_out+n_feats_sc, n_feats_out, ksize), # Shrink layer.
            act,
            base_block(conv, n_feats_out, n_feats_out, ksize, bias=not bn, bn=bn, act=act),
        ]

        self.upsampler = nn.Sequential(*m_upsample)
        self.layers = nn.Sequential(*m_layers)

    def forward(self, feats, sc):
        feats = self.upsampler(feats) # in: [B, n_feats_in, H/2, W/2], out: [B, n_feats_out, H, W]
        concat = torch.cat((sc, feats), dim=1) # out: [B, n_feats_sc+n_feats_out, H, W]

        feats = self.layers(concat)
        return feats

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            common.WABBlock(nn.Conv2d, n_feats=n_feat, kernel_size=kernel_size, dilates=4, bias=True, bn=False, act=act,
                    channel_attention=True, reduction=16, res_scale=res_scale) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size, padding=kernel_size//2, bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class DeblurUNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DeblurUNet, self).__init__()

        n_depth = args.n_depth
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        n_feats_in_d, n_feats_out_d = n_feats, n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = common.act_fn(args.act)
        self.n_depth = n_depth

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define enc module
        m_encoder = []
        for depth in range(n_depth): # 0, 1
            m_encoder.append(
                EncBlock(conv, nn.Identity, n_feats_in_d, n_feats_out_d, bn=False, act=act)
                )
            # prepare the next depth
            n_feats_in_d = n_feats_out_d
            n_feats_out_d = 2*n_feats_out_d


        # define body module
        m_body_exp = [conv(n_feats_in_d, n_feats_out_d, kernel_size)] # Expansion layer
        m_body = []
        for _ in range(n_resgroups):
            m_body.append(
                ResidualGroup(nn.Conv2d, n_feats_out_d, kernel_size, reduction=16, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks)
            )
        m_body.append(conv(n_feats_out_d, n_feats_out_d, kernel_size)) # Feature refinement layer
        # define dec module
        n_feats_in_d, n_feats_out_d = n_feats_out_d, n_feats_in_d

        m_decoder = []
        for depth in range(n_depth-1, -1, -1):
            m_decoder.append(
                DecBlock(conv, nn.Identity, n_feats_in_d, n_feats_out_d, n_feats_out_d, bn=False, act=act)
                )
            n_feats_in_d = n_feats_out_d
            n_feats_out_d //= 2

        # define tail module
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.encoder = nn.ModuleList(m_encoder)
        self.body_exp = nn.Sequential(*m_body_exp)
        self.body = nn.Sequential(*m_body)
        self.decoder = nn.ModuleList(m_decoder)
        self.tail = nn.Sequential(*m_tail)    
        
    def forward(self, x):
        if self.forward_ae_loss:
            return self.forward_enc_dec(x)
        else:
            return self.forward_whole_network(x)
    
    def forward_enc_dec(self, x):
        # Feature extraction
        x = self.head(x)
        feats = x

        # Encoder
        sc_encoder = []
        for depth in range(self.n_depth):
            x, sc = self.encoder[depth](x)
            sc_encoder.append(sc)

        # Body: Non-linear feature transform in the deepest level
        res = self.body_exp(x)
        
        # Decoder
        for depth in range(self.n_depth):
            res = self.decoder[depth](res, sc_encoder[self.n_depth-depth-1])

        # Feature global skip-connection.
        res = res + feats 
        # Reconstruction
        x = self.tail(res)

        return x       
    
    def forward_whole_network(self, x):
        # Feature extraction
        x = self.head(x)
        feats = x

        # Encoder
        sc_encoder = []
        for depth in range(self.n_depth):
            x, sc = self.encoder[depth](x)
            sc_encoder.append(sc)

        # Body: Non-linear feature transform in the deepest level
        x = self.body_exp(x)
        res = self.body(x)
        res += x
        
        # Decoder
        for depth in range(self.n_depth):
            res = self.decoder[depth](res, sc_encoder[self.n_depth-depth-1])

        # Feature global skip-connection.
        res = res + feats 
        # Reconstruction
        x = self.tail(res)

        return x         

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if (name.find('tail') == -1 and name.find('arbit') == -1):
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


