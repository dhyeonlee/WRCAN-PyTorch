def set_template(args):
    # Set the templates here
    if args.template.find('jpeg') >= 0:
        args.data_train = 'DIV2K_jpeg'
        args.data_test = 'DIV2K_jpeg'
        args.epochs = 200
        args.decay = '100'

    if args.template.find('EDSR_paper') >= 0:
        args.model = 'EDSR'
        args.n_resblocks = 32
        args.n_feats = 256
        args.res_scale = 0.1

    if args.template.find('MDSR') >= 0:
        args.model = 'MDSR'
        args.patch_size = 48
        args.epochs = 650

    if args.template.find('DDBPN') >= 0:
        args.model = 'DDBPN'
        args.patch_size = 128
        args.scale = '4'

        args.data_test = 'Set5'

        args.batch_size = 20
        args.epochs = 1000
        args.decay = '500'
        args.gamma = 0.1
        args.weight_decay = 1e-4

        args.loss = '1*MSE'

    if args.template.find('GAN') >= 0:
        args.epochs = 200
        if args.template.find('SRGAN') >= 0:
            args.lr = 1e-4
            args.decay = '100'
            args.gamma = 0.1
            args.model = 'SRResNet'
            args.patch_size = 96
            args.rgb_range = 1
            args.shift_mean = False
            args.normalization = 'batch'
            args.act = 'prelu'
        else:
            args.lr = 5e-5
            args.decay = '150'

    if args.template.find('RCAN') >= 0:
        args.model = 'RCAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True

    if args.template.find('VDSR') >= 0:
        args.model = 'VDSR'
        args.n_resblocks = 20
        args.n_feats = 64
        args.patch_size = 41
        args.lr = 1e-1

    if args.template.find('CPFR') >= 0:
        args.model = 'CPFR'
        args.n_resblocks = 6
        args.n_feats = 12
        args.patch_size = 21
        args.lr = 1e-1
        args.optimizer = 'SGD'
        args.n_colors = 1

    if args.template.lower().find('srresnet') >= 0:
        args.model = 'SRResNet'
        args.patch_size = 96
        args.epochs = 1000
        args.loss = '1*MSE'
        args.rgb_range = 1
        args.shift_mean = False
        args.normalization = 'batch'
        args.act = 'prelu'

