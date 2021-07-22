# -*- coding: utf-8 -*-

from typing import *
import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import numpy as np
import torch


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt', encoding='utf-8') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def check_parameters(module: torch.nn.Module):
    invalid_params = []
    for name, p in module.named_parameters():
        is_invalid = torch.isnan(p).any() or torch.isinf(p).any()
        if is_invalid:
            invalid_params.append(name)
    return invalid_params


def is_invalid(tensor: torch.Tensor):
    invalid = torch.isnan(tensor).any() or torch.isinf(tensor).any()
    return invalid


def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct
