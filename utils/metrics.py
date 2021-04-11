# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/11/2020 10:02 PM

from typing import *

import numpy as np
import pandas as pd


class AverageMetricTracker:
    def __init__(self, *keys, writer=None, fmt: Optional[str] = ':.6f'):
        '''
        Average metric tracker, can save multi-value
        :param keys: metrics
        :param writer:
        '''
        self.fmt = fmt
        self.writer = writer
        columns = ['total', 'counts', 'average', 'current_value']
        self._data = pd.DataFrame(np.zeros((len(keys), len(columns))), index=keys, columns=columns)
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.current_value[key] = value
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_multi_metrics(self, metrics: List[Dict[str, float]]):
        for metric in metrics:
            if 'n' not in metric.keys():
                metric['n'] = 1
            self.update(metric['key'], metric['value'], metric['n'])

    def avg(self, key):
        return self._data.average[key]

    def val(self, key):
        return self._data.current_value[key]

    def result(self):
        return dict(self._data.average)

    # def __str__(self):
    #     fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    #     return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_steps: int, meters: AverageMetricTracker, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_steps)
        self.meters = meters
        self.prefix = prefix
        self.fmtstr = '{name} {val:.6f} ({avg:.6f})'

    def display(self, step, keys: List):
        # prefix \t [step/num_steps] \t metric val: avg
        entries = [self.prefix + self.batch_fmtstr.format(step)]
        entries += [self.fmtstr.format(key, self.meters.val(key), self.meters.avg(key))
                    if key != '\n' else '\n'
                    for key in keys]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_steps):
        # [{:10d}/num_steps]
        num_digits = len(str(num_steps // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_steps) + ']'
