# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 8/13/2020 3:22 PM

import math
import argparse
import collections
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from model.master import MASTER
from parse_config import ConfigParser
import model.master as master_arch

import data_utils.datasets as master_dataset

from data_utils.datasets import ResizeWeight, TextDataset

from utils.label_util import LabelTransformer


def test_dataload():
    train_dataset = TextDataset(txt_file="/home/wwyu/data/OCRDATA/CUTE80/gt.txt",
                                img_root="/home/wwyu/data/OCRDATA/CUTE80",
                                transform=ResizeWeight((160, 48), gray_format=False), img_w=160,
                                img_h=48, convert_to_gray=False, split=' ')
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=8,
                                              drop_last=False)

    for idx, (img, label) in tqdm(enumerate(data_loader)):
        print(img.shape)
        print(len(label))
        break


def test_model():
    args = argparse.ArgumentParser(description='MASTER parameters')
    args.add_argument('-c', '--config', default='../configs/config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)

    logger = config.get_logger('train')

    model = config.init_obj('model_arch', master_arch)
    logger.info(model)

    x = torch.randn(2, 3, 48, 160)  # (bs, c, h, w)
    query = torch.ones(2, 100).long()  # (bs, len_tgt)
    output = model(x, query)  # (bs, len_tgt, n_cls)
    print(output.shape)  # (2, 100, n_cls)


def test_model_forward():
    # torch.backends.cudnn.benchmark = False
    args = argparse.ArgumentParser(description='MASTER parameters')
    args.add_argument('-c', '--config', default='../configs/config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to be available (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags default type target help')
    options = [
        CustomArgs(['-dist', '--distributed'], default='true', type=str, target='distributed',
                   help='run distributed training, true or false, (default: true).'
                        ' turn off distributed mode can debug code on one gpu/cpu'),
        CustomArgs(['--local_world_size'], default=1, type=int, target='local_world_size',
                   help='the number of processes running on each node, this is passed in explicitly '
                        'and is typically either $1$ or the number of GPUs per node. (default: 1)'),
        CustomArgs(['--local_rank'], default=0, type=int, target='local_rank',
                   help='this is automatically passed in via torch.distributed.launch.py, '
                        'process will be assigned a local rank ID in [0,local_world_size-1]. (default: 0)')
    ]

    config = ConfigParser.from_args(args, options)

    if torch.cuda.is_available() and config['local_rank'] != -1:
        torch.cuda.set_device(config['local_rank'])
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    model = config.init_obj('model_arch', master_arch)
    model.to(device)

    img_w = config['train_dataset']['args']['img_w']
    img_h = config['train_dataset']['args']['img_h']
    dataset = config.init_obj('train_dataset', master_dataset,
                              transform=ResizeWeight((img_w, img_h), gray_format=False),
                              convert_to_gray=False)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False)
    for idx, (img, label) in tqdm(enumerate(data_loader)):
        img = img.to(device)
        target = LabelTransformer.encode(label)
        target = target.to(device)
        target = target.permute(1, 0)
        output = model(img, target[:, :-1])
        print(output.shape) # (bs, len_tgt, n_cls)
        break


if __name__ == '__main__':
    # test_dataload()
    test_model()
    # test_model_forward()
