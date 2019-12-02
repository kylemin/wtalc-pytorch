from __future__ import print_function
import argparse
import os
import torch
from model import Model
from video_dataset import Dataset
from test import test
from train import train
from logger import Logger
import options
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.optim as optim
import numpy as np

if __name__ == '__main__':

    args = options.parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset_name = args.dataset_name
    if dataset_name == 'Thumos14reduced':
        dataset_name = 'THUMOS14'
    elif dataset_name == 'Thumos14':
        dataset_name = 'THUMOS14-whole'
    elif dataset_name == 'ActivityNet1.2':
        dataset_name = 'ActivityNet-1.2'

    exp_name = 'wtalc_%s_%d'%(dataset_name, args.seed)
    dataset = Dataset(args)
    if not os.path.exists('./ckpt/' + exp_name):
       os.makedirs('./ckpt/' + exp_name)
    if not os.path.exists('./logs/' + exp_name):
       os.makedirs('./logs/' + exp_name)
    logger = Logger('./logs/' + exp_name)

    model = Model(dataset.feature_size, dataset.num_class).to(device)

    if args.pretrained_ckpt is not None:
       model.load_state_dict(torch.load(args.pretrained_ckpt))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)

    for itr in range(1, args.max_iter+1):
       train(itr, dataset, args, model, optimizer, logger, device)
       if itr % 500 == 0:
          torch.save(model.state_dict(), './ckpt/' + exp_name + '/%s_%06d.pt' % (exp_name, itr))
          test(itr, dataset, args, model, logger, device)
