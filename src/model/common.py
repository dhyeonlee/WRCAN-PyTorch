import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def act_fn(act, num_parameters=1, negative_slope=0.01, inplace=True):
    if act == 'prelu':
        return nn.PReLU(num_parameters=num_parameters)
    elif act == 'lrelu':
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    else:
        return nn.ReLU(inplace=inplace)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, act=nn.ReLU(inplace=True)):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                act,
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class WABBlock(nn.Module):
    def __init__(
        self, conv=nn.Conv2d, 
        n_feats=256, kernel_size=3, dilates=4, stride=1,
        bias=True, bn=False, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), 
        channel_attention=False, reduction=16,
        res_scale=1.0):

        super(WABBlock, self).__init__()
        self.dilates = dilates
        self.stride = stride
        self.res_scale = res_scale

        m_dilated_parallel = []
        for i in range(dilates):
            m = []
            dilation = i+1
            padding = ((kernel_size-1)*dilation+1)//2
            m.append(conv(n_feats, n_feats//2, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            m.append(act)

            m_dilated_parallel.append(nn.Sequential(*m))

        m_shrink = [conv((n_feats//2)*dilates, n_feats, kernel_size=kernel_size, padding=kernel_size//2)]

        m_channel_attention = [CALayer(n_feats, reduction, act) if channel_attention else nn.Identity()]

        self.dilated_parallel = nn.ModuleList(m_dilated_parallel)
        self.shrink = nn.Sequential(*m_shrink)
        self.channel_attention = nn.Sequential(*m_channel_attention)

    def forward(self, x):
        res = []
        for i in range(self.dilates):
            res.append(self.dilated_parallel[i](x))

        res_cat = torch.cat(tuple(res), dim=1)
        res_cat = self.shrink(res_cat)

        res_cat = self.channel_attention(res_cat)
        
        res_cat = self.res_scale*res_cat ## residual_scaling
        
        if self.stride == 1:
            x = res_cat + x
        else:
            x = res_cat + F.avg_pool2d(x, 3, self.stride, padding=1)
        
        return x

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, n_feats_out=None, bn=False, act=False, bias=True):

        m = []
        if scale == 1:
            m.append(nn.Identity())
            
        elif (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4*n_feats if n_feats_out is None else 4*n_feats_out, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9*n_feats if n_feats_out is None else 9*n_feats_out, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

