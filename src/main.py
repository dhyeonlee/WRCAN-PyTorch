import cv2
import torch
import numpy as np
import random

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.n_GPUs > 1:
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)

checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            model = model.Model(args, checkpoint)
            loss_ = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, model, loss_, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
