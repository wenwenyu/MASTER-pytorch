# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 10/4/2020 14:18
# @Modified Time 1: 2021-05-18 by Novio

import numpy as np
import torch
from torch import nn

from model.backbone import ConvEmbeddingGC
from model.transformer import Encoder, Decoder


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, hidden_dim, vocab_size):
        """

        :param hidden_dim: dim of model
        :param vocab_size: size of vocabulary
        """
        super(Generator, self).__init__()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.fc(x)


class MASTER(nn.Module):
    """
     A standard Encoder-Decoder MASTER architecture.
    """

    def __init__(self, common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs):

        super(MASTER, self).__init__()
        self.with_encoder = common_kwargs['with_encoder']
        self.padding_symbol = 0
        self.sos_symbol = 1
        self.eos_symbol = 2
        self.build_model(common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs)
        for m_parameter in self.parameters():
            if m_parameter.dim() > 1:
                nn.init.xavier_uniform_(m_parameter)

    def build_model(self, common_kwargs, backbone_kwargs, encoder_kwargs, decoder_kwargs):
        target_vocabulary = common_kwargs['n_class']
        heads = common_kwargs['multiheads']
        dimensions = common_kwargs['model_size']
        self.conv_embedding_gc = ConvEmbeddingGC(
            gcb_kwargs=backbone_kwargs['gcb_kwargs'],
            in_channels=backbone_kwargs['in_channels']
        )
        # with encoder: cnn(+gc block) + transformer encoder + transformer decoder
        # without encoder: cnn(+gc block) + transformer decoder
        self.encoder = Encoder(
            _with_encoder=common_kwargs['with_encoder'],
            _multi_heads_count=heads,
            _dimensions=dimensions,
            _stacks=encoder_kwargs['stacks'],
            _dropout=encoder_kwargs['dropout'],
            _feed_forward_size=encoder_kwargs['feed_forward_size']
        )
        self.decoder = Decoder(
            _multi_heads_count=heads,
            _dimensions=dimensions,
            _stacks=decoder_kwargs['stacks'],
            _dropout=decoder_kwargs['dropout'],
            _feed_forward_size=decoder_kwargs['feed_forward_size'],
            _n_classes=target_vocabulary,
            _padding_symbol=self.padding_symbol,
        )
        self.generator = Generator(dimensions, target_vocabulary)

    def eval(self):
        self.conv_embedding_gc.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.generator.eval()

    def forward(self, _source, _target):
        embedding = self.conv_embedding_gc(_source)
        memory = self.encoder(embedding)
        output = self.decoder(_target, memory)
        return self.generator(output)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


if __name__ == '__main__':
    from parse_config import ConfigParser
    import json
    import argparse

    ag = argparse.ArgumentParser('Master Test Example')
    ag.add_argument('--config_path', type=str, required=True, help='配置文件地址')
    args = ag.parse_args()

    config_file_path = args.config_path
    with open(config_file_path, mode='r') as to_read_config_file:
        json_config = json.loads(to_read_config_file.read())
    config = ConfigParser(json_config)
    model = MASTER(**config['model_arch']['args'])
    model.eval()
    input_image_tensor = torch.zeros((1, 3, 32, 192), dtype=torch.float32)
    input_target_label_tensor = torch.zeros((1, 100), dtype=torch.long)
    with torch.no_grad():
        result = model(input_image_tensor, input_target_label_tensor).numpy()
        print(result.shape)
    traced_model = torch.jit.trace(model, (input_image_tensor, input_target_label_tensor))
    torch.jit.save(traced_model, 'master.pt')
    loaded_traced_model = torch.jit.load('master.pt')
    with torch.no_grad():
        loaded_model_result = loaded_traced_model(input_image_tensor, input_target_label_tensor).numpy()
        print(loaded_model_result.shape)
    print(np.mean(np.linalg.norm(result - loaded_model_result)))
