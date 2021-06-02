# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 10/4/2020 14:24
import collections
from pathlib import Path

import torch

STRING_MAX_LEN = 100
VOCABULARY_FILE_NAME = 'keys.txt'

class LabelConverterForMASTER:
    def __init__(self, classes, max_length=-1, ignore_over=False):
        """

        :param classes: alphabet(keys), key string or text vocabulary
        :param max_length:  max_length is mainly for controlling the statistics' stability,
         due to the layer normalisation. and the model can only predict max_length text.
         -1 is for fixing the length, the max_length will be computed dynamically for one batch.
         Empirically, it should be maximum length of text you want to predict.
        :param ignore_over:  (bool, default=False): whether or not to ignore over max length text.
        """

        cls_list = None
        if isinstance(classes, str):
            cls_list = list(classes)
        if isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('Key file is not found')
            with p.open(encoding='utf8') as f:
                classes = f.read()
                classes = classes.strip()
                cls_list = list(classes)
        elif isinstance(classes, list):
            cls_list = classes

        self.alphabet = cls_list
        self.alphabet_mapper = {'<EOS>': 1, '<SOS>': 2, '<PAD>': 0, '<UNK>': 3}
        # start of sequence
        # end of sequence
        for i, item in enumerate(self.alphabet):
            self.alphabet_mapper[item] = i + 4
        self.alphabet_inverse_mapper = {v: k for k, v in self.alphabet_mapper.items()}

        self.EOS = self.alphabet_mapper['<EOS>']
        self.SOS = self.alphabet_mapper['<SOS>']
        self.PAD = self.alphabet_mapper['<PAD>']
        self.UNK = self.alphabet_mapper['<UNK>']

        self.nclass = len(self.alphabet) + 4
        self.max_length = max_length
        self.ignore_over = ignore_over

    def encode(self, text):
        """ convert text to label index, add <SOS>, <EOS>, and do max_len padding
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.LongTensor targets:max_length × batch_size
        """
        if isinstance(text, str):
            text = [self.alphabet_mapper[item] if item in self.alphabet else self.UNK for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]  # encode

            if self.max_length == -1:
                local_max_length = max([len(x) for x in text])  # padding
                self.ignore_over = True
            else:
                local_max_length = self.max_length

            nb = len(text)

            targets = torch.zeros(nb, (local_max_length + 2))
            targets[:, :] = self.PAD

            for i in range(nb):

                if not self.ignore_over:
                    if len(text[i]) > local_max_length:
                        raise RuntimeError('Text is larger than {}: {}'.format(local_max_length, len(text[i])))

                targets[i][0] = self.SOS  # start
                targets[i][1:len(text[i]) + 1] = text[i]
                targets[i][len(text[i]) + 1] = self.EOS
            text = targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)

    def decode(self, t):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """

        # texts = list(self.dict.keys())[list(self.dict.values()).index(t)]
        if isinstance(t, torch.Tensor):
            texts = self.alphabet_inverse_mapper[t.item()]
        else:
            texts = self.alphabet_inverse_mapper[t]
        return texts


# LabelTransformer = strLabelConverterForTransformerWithVocabularyLevel(keys, max_length=STRING_MAX_LEN,
#                                                                       ignore_over=False)

LabelTransformer = LabelConverterForMASTER(Path(__file__).parent.joinpath(VOCABULARY_FILE_NAME),
                                           max_length=STRING_MAX_LEN, ignore_over=False)
