from collections import Counter

import torch
import torch.utils.data
from torch.utils.data import Sampler

from data_utils.datasets import TextDataset


class ImbalancedDatasetSampler(Sampler):
    def __init__(self, _data_source):
        # if indices is not provided, all elements in the dataset will be considered
        super().__init__(_data_source)
        self.num_samples = len(_data_source)

        all_labels = self._get_labels(_data_source)
        character_counter = Counter()
        total_characters = 0
        for m_label in all_labels:
            character_counter.update(m_label)
            total_characters += len(m_label)
        print('dataset character statistics')
        for m_label, m_count in sorted(character_counter.items(), key = lambda x: x[1]):
            print(f'char:{m_label},count:{m_count},ratio:{round(m_count * 100 / total_characters, 2)}%')
        weights = []
        for m_label in all_labels:
            weights.append(total_characters/(sum([character_counter.get(m_char) for m_char in m_label])/len(m_label)))
        self.weights = torch.DoubleTensor(weights)

    def _get_labels(self, _dataset):
        if isinstance(_dataset, TextDataset):
            return _dataset.get_all_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (i for i in torch.multinomial(self.weights, self.num_samples, replacement = True))

    def __len__(self):
        return self.num_samples
