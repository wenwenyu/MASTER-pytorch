# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 10/4/2020 14:18
# @Modified Time 1: 2021-05-18 by Novio

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential

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


class MultiInputSequential(Sequential):
    def forward(self, *_inputs):
        for m_module_index, m_module in enumerate(self):
            if m_module_index == 0:
                m_input = m_module(*_inputs)
            else:
                m_input = m_module(m_input)
        return m_input


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
        target_vocabulary = common_kwargs['n_class'] + 4
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
        self.encode_stage = nn.Sequential(self.conv_embedding_gc, self.encoder)
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
        self.decode_stage = MultiInputSequential(self.decoder, self.generator)

    def eval(self):
        self.conv_embedding_gc.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.generator.eval()

    def forward(self, _source, _target):
        encode_stage_result = self.encode_stage(_source)
        decode_stage_result = self.decode_stage(_target, encode_stage_result)
        return decode_stage_result

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


def predict(_memory, _source, _decode_stage, _max_length, _sos_symbol, _eos_symbol, _padding_symbol):
    batch_size = _source.size(0)
    device = _source.device
    to_return_label = \
        torch.ones((batch_size, _max_length + 2), dtype=torch.long).to(device) * _padding_symbol
    probabilities = torch.ones((batch_size, _max_length + 2), dtype=torch.float32).to(device)
    to_return_label[:, 0] = _sos_symbol
    for i in range(_max_length + 1):
        m_label = _decode_stage(to_return_label, _memory)
        m_probability = torch.softmax(m_label, dim=-1)
        m_max_probs, m_next_word = torch.max(m_probability, dim=-1)
        if m_next_word[:, i] == _eos_symbol:
            break
        to_return_label[:, i + 1] = m_next_word[:, i]
        probabilities[:, i + 1] = m_max_probs[:, i]
    return to_return_label, probabilities


if __name__ == '__main__':
    from parse_config import ConfigParser
    import json
    import os
    import argparse

    ag = argparse.ArgumentParser('Master Export Example')
    ag.add_argument('--config_path', type=str, required=True, help='配置文件地址')
    ag.add_argument('--checkpoint', type=str, required=False, help='训练好的模型的地址，没有的话就不加载')
    ag.add_argument('--target_directory', type=str, required=False, help='输出的pt文件的文件夹')
    ag.add_argument('--target_device', type=str, default='cuda:0', required=False, help='导出模型的设备')
    args = ag.parse_args()

    config_file_path = args.config_path
    device = args.target_device
    target_output_directory = args.target_directory
    os.makedirs(target_output_directory,exist_ok=True)
    with open(config_file_path, mode='r') as to_read_config_file:
        json_config = json.loads(to_read_config_file.read())
    config = ConfigParser(json_config)
    model = MASTER(**config['model_arch']['args'])
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')['model_state_dict']
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    input_image_tensor = torch.zeros((1, 3, 100, 150), dtype=torch.float32).to(device)
    input_target_label_tensor = torch.zeros((1, 100), dtype=torch.long).to(device)
    with torch.no_grad():
        encode_result = model.encode_stage(input_image_tensor)
    encode_stage_traced_model = torch.jit.trace(model.encode_stage, (input_image_tensor,))
    encode_traced_model_path = os.path.join(target_output_directory, 'master_encode.pt')
    torch.jit.save(encode_stage_traced_model, encode_traced_model_path)
    loaded_encode_stage_traced_model = torch.jit.load(encode_traced_model_path, map_location=device)
    with torch.no_grad():
        loaded_model_encode_result = loaded_encode_stage_traced_model(input_image_tensor, )
    print('encode diff',
          np.mean(np.linalg.norm(encode_result.cpu().numpy() - loaded_model_encode_result.cpu().numpy())))

    with torch.no_grad():
        decode_result = model.decode_stage(input_target_label_tensor, encode_result).cpu().numpy()
    decode_stage_traced_model = torch.jit.trace(model.decode_stage, (input_target_label_tensor, encode_result))
    decode_traced_model_path = os.path.join(target_output_directory, 'master_decode.pt')
    torch.jit.save(decode_stage_traced_model, decode_traced_model_path)
    loaded_decode_stage_traced_model = torch.jit.load(decode_traced_model_path, map_location=device)
    with torch.no_grad():
        loaded_model_decode_result = loaded_decode_stage_traced_model(
            input_target_label_tensor,
            encode_result,
        ).cpu().numpy()
    print('decode diff', np.mean(np.linalg.norm(decode_result - loaded_model_decode_result)))
    with torch.no_grad():
        model_label, model_label_prob = predict(encode_result, input_image_tensor, model.decode_stage, 10,2,1, 0)
        loaded_model_label, loaded_model_label_prob = predict(loaded_model_encode_result, input_image_tensor,
                                                              loaded_decode_stage_traced_model, 10,2, 1, 0)
        print(model_label.cpu().numpy(), model_label_prob.cpu().numpy())
        print(loaded_model_label.cpu().numpy(), loaded_model_label_prob.cpu().numpy())
