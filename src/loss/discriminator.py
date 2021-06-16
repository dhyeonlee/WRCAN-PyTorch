from model import common

import torch.nn as nn
import functools

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Discriminator(nn.Module):
    '''
        output is not normalized
    '''
    def __init__(self, args):
        super(Discriminator, self).__init__()

        in_channels = args.n_colors
        out_channels = 64
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(_out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        m_features = [_block(in_channels, out_channels)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(_block(in_channels, out_channels, stride=stride))

        patch_size = args.patch_size // (2**((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size**2, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]

        self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output

class Discriminator_patch(nn.Module):
    '''
        PatchGan's discriminator
        Ref.: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/f13aab8148bd5f15b9eb47b690496df8dadbab0c/models/networks.py#L538
    '''
    def __init__(self, args):
        super(Discriminator_patch, self).__init__()
        # arguments
        input_nc = args.n_colors
        ndf = 64
        n_layers = 3
        norm_layer = get_norm_layer(norm_type='batch')

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.features = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.features(input)


class Discriminator_kernel(nn.Module):
    ## Blind Super-Resolution Kernel Estimation using an Internal-GAN, NIPS2019
    def __init__(self, args):
        super(Discriminator_kernel, self).__init__()

        # First layer - Convolution (with no ReLU)
        self.first_layer = nn.utils.spectral_norm(nn.Conv2d(in_channels=args.n_colors, out_channels=64, kernel_size=7, padding=3, bias=True))
        feature_block = []  # Stacking layers with 1x1 kernels
        for _ in range(1, 6):
            feature_block += [nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, bias=True)),
                              nn.BatchNorm2d(64),
                              nn.ReLU(True)]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, bias=True)),
                                         nn.Sigmoid())


    def forward(self, input_tensor):
        receptive_extraction = self.first_layer(input_tensor)
        features = self.feature_block(receptive_extraction)
        return self.final_layer(features)


class Discriminator_RP(nn.Module):
    '''
        output is not normalized
        This discriminator architecture follows the architecture in RPSRGAN_IEEE_Access_2019.
        However, dropout is removed because it has the same purpose as BN.
    '''
    def __init__(self, args):
        super(Discriminator_RP, self).__init__()

        in_channels = args.n_colors
        out_channels = 256
        depth = 7

        def _block(_in_channels, _out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(
                    _in_channels,
                    _out_channels,
                    3,
                    padding=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(_out_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        m_features = [_block(in_channels, out_channels)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 0:
                out_channels //= 2
            m_features += [
                nn.Dropout2d(p=0.4),
                _block(in_channels, out_channels)
            ]

        m_classifier = [
            nn.Dropout(p=0.4),
            nn.Linear(out_channels*args.patch_size**2, 1)
        ]

        self.features = nn.Sequential(*m_features)
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))

        return output