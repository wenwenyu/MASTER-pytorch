# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/12/2020 11:29 PM

import os
import argparse
import collections
import random

import numpy as np
import torch
import torch.distributed as dist

import model.master as master_arch
import data_utils.datasets as master_dataset
from data_utils.datasets import ResizeWeight
from data_utils.datasets import DistValSampler, DistCollateFn

from parse_config import ConfigParser
from trainer import Trainer


def main(config: ConfigParser, local_master: bool, logger=None):
    train_batch_size = config['trainer']['train_batch_size']
    val_batch_size = config['trainer']['val_batch_size']

    train_num_workers = config['trainer']['train_num_workers']
    val_num_workers = config['trainer']['val_num_workers']

    # setup  dataset and data_loader instances
    img_w = config['train_dataset']['args']['img_w']
    img_h = config['train_dataset']['args']['img_h']
    in_channels = config['model_arch']['args']['backbone_kwargs']['in_channels']
    convert_to_gray = False if in_channels == 3 else True
    train_dataset = config.init_obj('train_dataset', master_dataset,
                                    transform=ResizeWeight((img_w, img_h), gray_format=convert_to_gray),
                                    convert_to_gray=convert_to_gray)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) \
        if config['distributed'] else None

    is_shuffle = False if config['distributed'] else True
    train_data_loader = config.init_obj('train_loader', torch.utils.data.dataloader,
                                        dataset=train_dataset,
                                        sampler=train_sampler,
                                        batch_size=train_batch_size,
                                        collate_fn=DistCollateFn(training=True),
                                        num_workers=train_num_workers,
                                        shuffle=is_shuffle)

    val_dataset = config.init_obj('val_dataset', master_dataset,
                                  transform=ResizeWeight((img_w, img_h), gray_format=convert_to_gray),
                                  convert_to_gray=convert_to_gray)
    val_sampler = DistValSampler(list(range(len(val_dataset))), batch_size=val_batch_size,
                                 distributed=config['distributed'])
    val_data_loader = config.init_obj('val_loader', torch.utils.data.dataloader,
                                      dataset=val_dataset,
                                      batch_sampler=val_sampler,
                                      batch_size=1,
                                      collate_fn=DistCollateFn(training=True),
                                      num_workers=val_num_workers)

    logger.info(f'Dataloader instances have finished. Train datasets: {len(train_dataset)} '
                f'Val datasets: {len(val_dataset)} Train_batch_size/gpu: {train_batch_size} '
                f'Val_batch_size/gpu: {val_batch_size}.') if local_master else None

    max_len_step = len(train_data_loader)
    if config['trainer']['max_len_step'] is not None:
        max_len_step = min(config['trainer']['max_len_step'], max_len_step)

    # build model architecture
    model = config.init_obj('model_arch', master_arch)
    logger.info(f'Model created, trainable parameters: {model.model_parameters()}.') if local_master else None

    # build optimizer, learning rate scheduler.
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    if config['lr_scheduler']['type'] is not None:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None
    logger.info('Optimizer and lr_scheduler created.') if local_master else None

    # log training related information
    logger.info('Max_epochs: {} Log_step_interval: {} Validation_step_interval: {}.'.
                format(config['trainer']['epochs'],
                       config['trainer']['log_step_interval'],
                       config['trainer']['val_step_interval'])) if local_master else None

    logger.info('Training start...') if local_master else None

    trainer = Trainer(model, optimizer, config,
                      data_loader=train_data_loader,
                      valid_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler,
                      max_len_step=max_len_step)

    trainer.train()

    logger.info('Distributed training end...') if local_master else None


def entry_point(config: ConfigParser):
    '''
    entry-point function for a single worker distributed training
    a single worker contain (torch.cuda.device_count() / local_world_size) gpus
    '''

    local_world_size = config['local_world_size']

    # check distributed environment cfgs
    if config['distributed']:  # distributed gpu mode
        # check gpu available
        if torch.cuda.is_available():
            if torch.cuda.device_count() < local_world_size:
                raise RuntimeError(f'the number of GPU ({torch.cuda.device_count()}) is less than '
                                   f'the number of processes ({local_world_size}) running on each node')
            local_master = (config['local_rank'] == 0)
        else:
            raise RuntimeError('CUDA is not available, Distributed training is not supported.')
    else:  # one gpu or cpu mode
        if config['local_world_size'] != 1:
            raise RuntimeError('local_world_size must set be to 1, if distributed is set to false.')
        config.update_config('local_rank', 0)
        local_master = True
        config.update_config('global_rank', 0)

    logger = config.get_logger('train') if local_master else None
    if config['distributed']:
        logger.info('Distributed GPU training model start...') if local_master else None
    else:
        logger.info('One GPU or CPU training mode start...') if local_master else None
    # else:
    #     sys.stdin.close()

    # cfg CUDNN whether deterministic
    if config['deterministic']:
        fix_random_seed_for_reproduce(config['seed'])
        logger.warn('You have chosen to deterministic training. '
                    'This will fix random seed, turn on the CUDNN deterministic setting, turn off the CUDNN benchmark '
                    'which can slow down your training considerably! '
                    ) if local_master else None
    else:
        torch.backends.cudnn.benchmark = True
        logger.warn('You have chosen to benchmark training. '
                    'This will turn on the CUDNN benchmark setting'
                    'which can speed up your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints due to RandomizedMultiLinearMap need deterministic turn on.'
                    ) if local_master else None

    if config['distributed']:
        # init process group
        dist.init_process_group(backend='nccl', init_method='env://')
        config.update_config('global_rank', dist.get_rank())
        # log distributed training cfg
        logger.info(
            f'[Process {os.getpid()}] world_size = {dist.get_world_size()}, '
            + f'rank = {dist.get_rank()}, backend={dist.get_backend()}'
        ) if local_master else None

    # start train
    main(config, local_master, logger if local_master else None)
    if config['distributed']:
        # tear down the process group
        dist.destroy_process_group()


def fix_random_seed_for_reproduce(seed):
    # fix random seeds for reproducibility,
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # for current gpu
    torch.cuda.manual_seed_all(seed)  # for all gpu
    torch.backends.cudnn.benchmark = False  # if benchmark=True, speed up training, and deterministic will set be False
    torch.backends.cudnn.deterministic = True  # which can slow down training considerably


def parse_args():
    global config
    args = argparse.ArgumentParser(description='MASTER PyTorch Distributed Training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to be available (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags default type target help')
    options = [
        # CustomArgs(['--lr', '--learning_rate'], default=0.0001, type=float, target='optimizer;args;lr',
        #            help='learning rate (default: 0.0001)'),
        # CustomArgs(['-dist', '--distributed'], default='true', type=str, target='distributed',
        #            help='run distributed training, true or false, (default: true).'
        #                 ' turn off distributed mode can debug code on one gpu/cpu'),
        # CustomArgs(['--local_world_size'], default=1, type=int, target='local_world_size',
        #            help='the number of processes running on each node, this is passed in explicitly '
        #                 'and is typically either $1$ or the number of GPUs per node. (default: 1)'),
        # CustomArgs(['--local_rank'], default=0, type=int, target='local_rank',
        #            help='this is automatically passed in via torch.distributed.launch.py, '
        #                 'process will be assigned a local rank ID in [0,local_world_size-1]. (default: 0)'),
        CustomArgs(['--finetune'], default='false', type=str, target='finetune',
                   help='finetune mode will load resume checkpoint, but do not use previous config and optimizer '
                        '(default: false), so there has three running mode: normal, resume, finetune')
    ]
    config = ConfigParser.from_args(args, options)
    return config


if __name__ == '__main__':
    config = parse_args()
    # The main entry point is called directly without using subprocess, called by torch.distributed.launch.py
    entry_point(config)
